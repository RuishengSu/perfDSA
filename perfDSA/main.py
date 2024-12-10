import argparse
import glob
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from autotici.utils.utils import minip
from natsort import natsorted
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import cm
from perfDSA.path_utils import extract_patient_id_from_path
from scipy import ndimage
from scipy.interpolate import interp1d
import cv2 as cv

import pydicom
import deconvolution
import dicom_reader.reader as dicom_reader
import dicom_utils
import elastix
from dataframe.DSABase import Pre_Post_EVT_Enum
from dataframe.sequence_repository import SequenceRepository
from perfDSA import config
from perfDSA.utils import *

logger = logging.getLogger(__name__)


def longest(binary_mat):
    mat = np.array(binary_mat, dtype=int)
    for i in range(1, mat.shape[0]):
        mat[i] += np.multiply(mat[i], mat[i - 1])
    end_idx = np.argmax(mat, axis=0)
    duration = np.max(mat, axis=0)
    start_idx = end_idx - duration + 1
    return start_idx, end_idx, duration


def save_fig(fig_save_path, *img):
    Path(fig_save_path).parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(fig_save_path, np.concatenate([*img]))


def truncate(img, img_min=0, img_max=255):
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img


def _get_patient_sequence(patient_path, view):
    df_sequence = SequenceRepository().get_dataframe()
    patient_id = extract_patient_id_from_path(patient_path)
    preEVT_sequence_name = df_sequence[(df_sequence['patient_id'] == patient_id)
                                       & (df_sequence['view'] == view)
                                       & (df_sequence['pre_post_EVT'] == Pre_Post_EVT_Enum.PRE_EVT.value)][
        'sequence_name'].iloc[0]
    preEVT_sequence_path = os.path.join(patient_path, preEVT_sequence_name + '.dcm')
    ds_preEVT = dicom_reader.read_file(preEVT_sequence_path, defer_size="1 KB", stop_before_pixels=False)
    cum_time_preEVT = np.cumsum(ds_preEVT.FrameTimeVector)
    preEVT_sequence = ds_preEVT.pixel_array
    # preEVT_sequence = np.transpose(preEVT_sequence, (2, 1, 0))

    postEVT_sequence_name = df_sequence[(df_sequence['patient_id'] == patient_id)
                                        & (df_sequence['view'] == view)
                                        & (df_sequence['pre_post_EVT'] == Pre_Post_EVT_Enum.POST_EVT.value)][
        'sequence_name'].iloc[0]
    postEVT_sequence_path = os.path.join(patient_path, postEVT_sequence_name + '.dcm')
    ds_postEVT = dicom_reader.read_file(postEVT_sequence_path, defer_size="1 KB", stop_before_pixels=False)
    cum_time_postEVT = np.cumsum(ds_postEVT.FrameTimeVector)
    postEVT_sequence = ds_postEVT.pixel_array
    # postEVT_sequence = np.transpose(postEVT_sequence, (2, 1, 0))

    return preEVT_sequence, postEVT_sequence, cum_time_preEVT, cum_time_postEVT, preEVT_sequence_name, postEVT_sequence_name


def motion_correction(preEVT_sequence, postEVT_sequence, patient_id, view):
    """Perform motion correction on the DSA sequences. Output image dtype: np.float32"""
    pickle_path = os.path.join(config.latest_mc_sequence_dirpath, '{}_{}.pickle'.format(patient_id, view))
    if config.reuse_motion_correction_results and os.path.isfile(pickle_path):
        logger.info("Reusing existing motion correction results")
        with open(pickle_path, 'rb') as f:
            preEVT_sequence, postEVT_sequence = pickle.load(f)
    else:
        logger.info("Correcting sequence motion artifacts")
        preEVT_sequence = elastix.elastix_align_sequence(preEVT_sequence)
        postEVT_sequence = elastix.elastix_align_sequence(postEVT_sequence)
        with open(pickle_path, 'wb') as f:
            pickle.dump([preEVT_sequence, postEVT_sequence], f)

    return preEVT_sequence, postEVT_sequence


def sequence_registration(fixed_sequence, moving_sequence, patient_id, view, moving_mask=None):
    """Perform sequence minip registration to register the moving sequence to the fixed sequence"""
    minip_fixed = minip(fixed_sequence)
    minip_moving = minip(moving_sequence)
    _, transform_parameters = elastix.elastix_align_image_pair(sitk.GetImageFromArray(minip_fixed),
                                                               sitk.GetImageFromArray(minip_moving))
    moving_sequence = elastix.elastix_transform_sequence(moving_sequence, transform_parameters)
    moving_mask = elastix.elastix_transform_image(moving_mask, transform_parameters)
    moving_mask = moving_mask > 0.5

    vis1 = np.concatenate([minip_moving, minip_fixed], axis=1)
    vis2 = np.concatenate([minip(moving_sequence), minip_fixed], axis=1)
    vis_save_path = os.path.join(config.latest_vis_sequence_registration_dirpath, "{}_{}.png".format(patient_id, view))
    save_fig(vis_save_path, vis1, vis2)
    vis_save_path = os.path.join(config.current_vis_sequence_registration_dirpath, "{}_{}.png".format(patient_id, view))
    save_fig(vis_save_path, vis1, vis2)
    return moving_sequence, moving_mask


def pre_post_registration(preEVT_sequence, postEVT_sequence, patient_id, view):
    minip_preEVT = minip(preEVT_sequence)
    minip_postEVT = minip(postEVT_sequence)

    '''Perform motion correction on the DSA sequences. Output image dtype: np.float32'''
    pickle_path = os.path.join(config.latest_pre_post_registration_dirpath, '{}_{}.pickle'.format(patient_id, view))
    if config.reuse_pre_post_registration_results and os.path.isfile(pickle_path):
        logger.info("Reusing pre- and post-EVT registration results")
        with open(pickle_path, 'rb') as f:
            minip_preEVT, transformation_matrix = pickle.load(f)
    else:
        logger.info("Registering preEVT minip to postEVT")
        minip_preEVT, transformation_matrix = elastix.elastix_align_image_pair(
            sitk.GetImageFromArray(minip_postEVT),
            sitk.GetImageFromArray(minip_preEVT))
        # with open(pickle_path, 'wb') as f:
        #     pickle.dump([minip_preEVT, transformation_matrix], f)
    return minip_preEVT, transformation_matrix


def convert_to_enhancement_sequence(sequence):
    backgound = np.median(sequence)
    max_density_map = np.max(sequence, axis=0)
    max_density_map[max_density_map > backgound] = backgound
    min_density_map = np.min(sequence, axis=0)
    max_diff = np.max(max_density_map - min_density_map)
    return 255 * (max_density_map - sequence) / max_diff


def me(sequence):
    backgound = np.median(sequence)
    max_density_map = np.max(sequence, axis=0)
    max_density_map[max_density_map > backgound] = backgound
    min_density_map = np.min(sequence, axis=0)
    me_map = max_density_map - min_density_map
    normalized_me_map = (me_map - np.min(me_map)) / (np.max(me_map) - np.min(me_map))  # convert index to 0-1 range
    # me_map = 255 * me_map  # convert index to 0-255 range

    '''generate colormap'''
    me_colormap = np.uint8((cm.binary(normalized_me_map)[:, :, :3]) * 255)
    # me_colormap[me_map <= config.noise_level] = [255, 255, 255]  # whiten the background pixels
    return me_map, normalized_me_map, me_colormap


def paired_me(sequence_preEVT, sequence_postEVT):
    """
    maximum enhancement
    :param sequence_preEVT: a tx1024x1024 DSA series, preEVT
    :param sequence_postEVT: a tx1024x1024 DSA series, postEVT
    """

    logger.info("Computing maximum enhancement colormap")
    me_map_preEVT = me(sequence_preEVT)
    me_map_postEVT = me(sequence_postEVT)

    '''diff'''
    me_diff = me_map_postEVT - me_map_preEVT
    me_diff = me_diff / np.max(abs(me_diff))
    me_diff = 0.5 * (1 + me_diff)

    '''generate colormap'''
    me_map_preEVT = np.uint8((cm.binary(me_map_preEVT)[:, :, :3]) * 255)
    me_map_postEVT = np.uint8((cm.binary(me_map_postEVT)[:, :, :3]) * 255)
    me_diff_colormap = np.uint8((cm.PiYG(me_diff)[:, :, :3]) * 255)

    label_image = cv.putText(img=np.zeros((100, *me_map_preEVT.shape[1:])), text="Density", org=(350, 70),
                             fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
    me_map_preEVT = np.concatenate([label_image, me_map_preEVT])
    return me_map_preEVT, me_map_postEVT, me_diff_colormap


def tmax(density_sequence, irf_sequence, min_sequence_length):
    peak_map = 0.9 * np.max(irf_sequence, axis=0)  # to avoid instability caused by flat peak
    peak_index_map = irf_sequence - peak_map > 0
    tmax_map = np.argmax(peak_index_map, axis=0)
    tmax_map = tmax_map / (min_sequence_length - 1)

    me_map = np.max(density_sequence, axis=0)
    # tmax_map[me_map <= config.noise_level] = 1

    tmax_colormap = np.uint8((cm.jet(tmax_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet or RdBu_r
    tmax_colormap[me_map <= config.noise_level] = [255, 255, 255]  # whiten pixels if no significant density change
    return tmax_colormap, tmax_map, me_map


def paired_tmax(density_sequence_preEVT, density_sequence_postEVT, irf_sequence_preEVT, irf_sequence_postEVT):
    """
    calculates a parametric image based on time to peak
    :param density_sequence_preEVT: a tx1024x1024 DSA series, preEVT
    :param density_sequence_postEVT: a tx1024x1024 DSA series, postEVT
    :param irf_sequence_preEVT: a tx1024x1024 DSA series, preEVT
    :param irf_sequence_postEVT: a tx1024x1024 DSA series, postEVT
    """

    logger.info("Computing Tmax colormap")
    min_sequence_len = min(irf_sequence_preEVT.shape[0], irf_sequence_postEVT.shape[0])
    tmax_colormap_preEVT, tmax_map_preEVT, me_map_preEVT = _tmax(density_sequence_preEVT, irf_sequence_preEVT,
                                                                 min_sequence_len)
    tmax_colormap_postEVT, tmax_map_postEVT, me_map_postEVT = _tmax(density_sequence_postEVT, irf_sequence_postEVT,
                                                                    min_sequence_len)

    '''diff'''
    tmax_diff = tmax_map_preEVT - tmax_map_postEVT
    tmax_diff[(me_map_preEVT <= config.noise_level) & (me_map_postEVT <= config.noise_level)] = 0
    normalized_tmax_diff = tmax_diff / np.amax(abs(tmax_diff))
    normalized_tmax_diff = 0.5 * (1 + normalized_tmax_diff)
    tmax_diff_colormap = np.uint8((cm.PiYG(normalized_tmax_diff)[:, :, :3]) * 255)  # convert to colormap, cm: RdYlGn

    label_image = cv.putText(img=np.zeros((100, *tmax_colormap_preEVT.shape[1:])), text="Tmax", org=(350, 70),
                             fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
    tmax_colormap_preEVT = np.concatenate([label_image, tmax_colormap_preEVT])

    logger.info("----> preEVT Tmax max: {}, min: {}".format(np.max(tmax_map_preEVT), np.min(tmax_map_preEVT)))
    logger.info("---> postEVT Tmax max: {}, min: {}".format(np.max(tmax_map_postEVT), np.min(tmax_map_postEVT)))
    logger.info("---> diffEVT Tmax max: {}, min: {}".format(np.max(tmax_diff), np.min(tmax_diff)))

    return tmax_colormap_preEVT, tmax_colormap_postEVT, tmax_diff_colormap


def rCBF(density_sequence_preEVT, density_sequence_postEVT, irf_sequence_preEVT, irf_sequence_postEVT):
    """
    calculates a parametric image based on time to peak
    :param density_sequence_preEVT: a tx1024x1024 DSA series, preEVT
    :param density_sequence_postEVT: a tx1024x1024 DSA series, postEVT
    :param irf_sequence_preEVT: a tx1024x1024 DSA series, preEVT
    :param irf_sequence_postEVT: a tx1024x1024 DSA series, postEVT
    """

    def _cbf(density_sequence, irf_sequence):
        cbf_map = np.max(irf_sequence, axis=0)  # to avoid instability caused by flat peak
        logger.info("CBF max: {}".format(np.max(cbf_map)))
        cbf_map = cbf_map / np.max(cbf_map)

        me_map = np.max(density_sequence, axis=0)
        # cbf_map[me_map <= config.noise_level] = 1

        cbf_colormap = np.uint8((cm.jet(cbf_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet or RdBu_r
        cbf_colormap[me_map <= config.noise_level] = [255, 255, 255]  # whiten pixels if no significant density change
        return cbf_colormap, cbf_map, me_map

    logger.info("Computing time to peak colormap")
    cbf_colormap_preEVT, cbf_preEVT, me_map_preEVT = _cbf(density_sequence_preEVT, irf_sequence_preEVT)
    cbf_colormap_postEVT, cbf_postEVT, me_map_postEVT = _cbf(density_sequence_postEVT, irf_sequence_postEVT)

    '''diff'''
    cbf_diff = cbf_postEVT - cbf_preEVT
    normalized_cbf_diff = cbf_diff / np.amax(abs(cbf_diff))
    normalized_cbf_diff = 0.5 * (1 + normalized_cbf_diff)
    cbf_diff_colormap = np.uint8((cm.PiYG(normalized_cbf_diff)[:, :, :3]) * 255)  # convert to colormap, cm: RdYlGn

    label_image = cv.putText(img=np.zeros((100, *cbf_colormap_preEVT.shape[1:])), text="CBF", org=(350, 70),
                             fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
    cbf_colormap_preEVT = np.concatenate([label_image, cbf_colormap_preEVT])
    return cbf_colormap_preEVT, cbf_colormap_postEVT, cbf_diff_colormap, cbf_preEVT, cbf_postEVT


def toa(density_sequence):
    me_map = np.max(density_sequence, axis=0)
    uptake_start_density_map = 0.2 * me_map
    toa_index_map = (density_sequence - uptake_start_density_map) > 0
    toa_map = np.argmax(toa_index_map, axis=0)
    toa_map = toa_map / (density_sequence.shape[0] - 1)  # convert index to 0-1 range
    # toa_map[toa_map > 1] = 1
    toa_map[me_map <= config.noise_level] = 0
    normalized_toa_map = (toa_map - np.min(toa_map)) / (np.max(toa_map) - np.min(toa_map))  # convert index to 0-1 range
    toa_colormap = np.uint8((cm.jet_r(normalized_toa_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet
    # toa_colormap[me_map <= config.noise_level] = [255, 255, 255]  # whiten the background pixels

    return toa_map, normalized_toa_map, toa_colormap


def ttp(density_sequence):
    me_map = np.max(density_sequence, axis=0)
    peak_map = 0.9 * me_map  # to avoid instability caused by flat peak
    peak_index_map = (density_sequence - peak_map) > 0
    ttp_map = np.argmax(peak_index_map, axis=0)
    ttp_map = ttp_map / (density_sequence.shape[0] - 1)
    # ttp_map[ttp_map > 1] = 1
    ttp_map[me_map <= config.noise_level] = 0
    normalized_ttp_map = (ttp_map - np.min(ttp_map)) / (np.max(ttp_map) - np.min(ttp_map))  # convert index to 0-1 range
    ttp_colormap = np.uint8((cm.jet_r(normalized_ttp_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet
    # ttp_colormap[me_map <= config.noise_level] = [255, 255, 255]  # whiten the background pixels

    return ttp_map, normalized_ttp_map, ttp_colormap


def toa_ttp_wir(density_sequence):
    max_map = np.max(density_sequence, axis=0)
    min_map = np.min(density_sequence, axis=0)

    # toa
    uptake_start_density_map = min_map + 0.2 * (max_map - min_map)
    toa_index_array = np.array((density_sequence - uptake_start_density_map) > 0)
    arrival_start, arrival_end, arrival_duration = longest(toa_index_array)
    toa_map = arrival_start
    toa_map[max_map <= config.noise_level] = 0

    # ttp
    peak_map = max_map - 0.1 * (max_map - min_map)  # to avoid instability caused by flat peak
    peak_index_array = np.array((density_sequence - peak_map) > 0)
    peak_start, peak_end, peak_duration = longest(peak_index_array)
    # ttp_map = (peak_start + peak_end) // 2
    ttp_map = peak_start
    ttp_map[max_map - min_map <= config.noise_level] = 0

    # relationship checks
    toa_map[peak_start <= arrival_start + 5] = 0
    toa_map[peak_end > arrival_end] = 0
    toa_map[arrival_duration < 15 * 2] = 0
    toa_map[peak_start > arrival_start + 15 * 10] = 0

    ttp_map[peak_start <= arrival_start + 5] = 0
    ttp_map[peak_end > arrival_end] = 0
    ttp_map[peak_duration < 2] = 0
    ttp_map[peak_start > arrival_start + 15 * 10] = 0

    # toa_map[arrival_duration > 15*20] = 0
    # ttp_map[peak_duration > 15*15] = 0

    # wir
    wir_map = ttp_map - toa_map
    wir_map[wir_map < 0] = 0
    # wir_map[wir_map > 15*6] = 0
    wir_map[wir_map > 0] = ((peak_map - uptake_start_density_map)[wir_map > 0]) / ((ttp_map - toa_map)[wir_map > 0])

    # generating colormap
    # toa_map = toa_map / (density_sequence.shape[0] - 1)  # convert index to 0-1 range
    normalized_toa_map = (toa_map - np.min(toa_map)) / (np.max(toa_map) - np.min(toa_map))  # convert index to 0-1 range
    toa_colormap = np.uint8((cm.jet_r(normalized_toa_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet
    # toa_colormap[me_map <= config.noise_level] = [255, 255, 255]  # whiten the background pixels

    # ttp_map = ttp_map / (density_sequence.shape[0] - 1)
    normalized_ttp_map = (ttp_map - np.min(ttp_map)) / (np.max(ttp_map) - np.min(ttp_map))  # convert index to 0-1 range
    ttp_colormap = np.uint8((cm.jet_r(normalized_ttp_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet

    normalized_wir_map = (wir_map - np.min(wir_map)) / (np.max(wir_map) - np.min(wir_map))  # convert index to 0-1 range
    wir_colormap = np.uint8((cm.jet_r(normalized_wir_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet

    return toa_colormap, ttp_colormap, wir_colormap


def wir(density_sequence):
    """
    wash in rate
    :param density_sequence: a tx1024x1024 DSA series
    """

    me_map = np.max(density_sequence, axis=0)
    min_map = np.min(density_sequence, axis=0)
    toa_map, normalized_toa_map, toa_colormap = toa(density_sequence)
    ttp_map, normalized_ttp_map, ttp_colormap = ttp(density_sequence)

    uptake_start_density_map = min_map + 0.2 * (me_map - min_map)
    peak_map = min_map + 0.9 * (me_map - min_map)  # to avoid instability caused by flat peak

    if np.min(ttp_map - toa_map) < 0:
        raise ValueError("TTP is earlier than TOA! please double check!")
    if np.min(peak_map - uptake_start_density_map) < 0:
        raise ValueError("peak density is smaller than startup density! please double check!")
    wir_map = ttp_map - toa_map
    wir_map[wir_map > 0] = ((peak_map - uptake_start_density_map)[wir_map > 0]) / ((ttp_map - toa_map)[wir_map > 0])
    # wir_map[(wir_map > 100) | (wir_map < 0)] = 0

    normalized_wir_map = (wir_map - np.min(wir_map)) / (
            np.max(wir_map) - np.min(wir_map))  # convert index to 0-1 range
    wir_colormap = np.uint8((cm.jet_r(normalized_wir_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet
    # wir_colormap[me_map <= config.noise_level] = [255, 255, 255]  # whiten the background pixels
    return wir_map, normalized_wir_map, wir_colormap


def paired_wir(density_sequence_preEVT, density_sequence_postEVT):
    """
    wash in rate
    :param density_sequence_preEVT: a tx1024x1024 DSA series, preEVT
    :param density_sequence_postEVT: a tx1024x1024 DSA series, postEVT
    """

    def _wir(density_sequence, min_sequence_length):
        me_map = np.max(density_sequence, axis=0)

        peak_map = 0.9 * me_map  # to avoid instability caused by flat peak
        peak_index_map = density_sequence - peak_map > 0
        ttp_map = np.argmax(peak_index_map, axis=0)
        ttp_map = ttp_map / min_sequence_length
        ttp_map[ttp_map > 1] = 1
        ttp_map[me_map <= config.noise_level] = 1

        uptake_start_density_map = 0.1 * me_map
        toa_index_map = density_sequence - uptake_start_density_map > 0
        toa_map = np.argmax(toa_index_map, axis=0)
        toa_map = toa_map / (min_sequence_length - 1)  # convert index to 0-1 range
        toa_map[toa_map > 1] = 1
        toa_map[me_map <= config.noise_level] = 1

        if np.min(ttp_map - toa_map) < 0:
            raise ValueError("TTP is earlier than TOA! please double check!")
        if np.min(peak_map - uptake_start_density_map) < 0:
            raise ValueError("peak density is smaller than startup density! please double check!")
        wir_map = ttp_map - toa_map
        wir_map[wir_map > 0] = ((peak_map - uptake_start_density_map)[wir_map > 0]) / ((ttp_map - toa_map)[wir_map > 0])
        # wir_map[(wir_map > 100) | (wir_map < 0)] = 0

        normalized_wir_map = (wir_map - np.min(wir_map)) / (
                np.max(wir_map) - np.min(wir_map))  # convert index to 0-1 range
        wir_colormap = np.uint8((cm.jet_r(normalized_wir_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet
        wir_colormap[me_map <= config.noise_level] = [255, 255, 255]  # whiten the background pixels
        return wir_colormap, normalized_wir_map, wir_map, me_map

    logger.info("Computing wash in rate colormap")
    min_sequence_len = min(density_sequence_preEVT.shape[0], density_sequence_postEVT.shape[0])
    wir_colormap_preEVT, normalized_wir_map_preEVT, wir_map_preEVT, me_map_preEVT = _wir(density_sequence_preEVT,
                                                                                         min_sequence_len)
    wir_colormap_postEVT, normalized_wir_map_postEVT, wir_map_postEVT, me_map_postEVT = _wir(
        density_sequence_postEVT, min_sequence_len)

    '''diff'''
    wir_diff = wir_map_postEVT - wir_map_preEVT
    wir_diff[(me_map_preEVT <= config.noise_level) & (me_map_postEVT <= config.noise_level)] = 0
    wir_diff = wir_diff / np.max(abs(wir_diff))
    wir_diff = 0.5 * (1 + wir_diff)
    wir_diff_colormap = np.uint8((cm.PiYG(wir_diff)[:, :, :3]) * 255)

    label_image = cv.putText(img=np.zeros((100, *wir_colormap_preEVT.shape[1:])), text="WIR", org=(350, 70),
                             fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
    wir_colormap_preEVT = np.concatenate([label_image, wir_colormap_preEVT])
    return wir_colormap_preEVT, wir_colormap_postEVT, wir_diff_colormap


def boe(density_sequence):
    me_map = np.max(density_sequence, axis=0)
    min_map = np.min(density_sequence, axis=0)
    thres_density_map = min_map + 0.8 * (me_map - min_map)
    boe_index_array = np.array((density_sequence - thres_density_map) >= 0)
    boe_start, boe_end, boe_duration = longest(boe_index_array)
    boe_map = boe_duration
    boe_map[me_map - min_map <= config.noise_level] = 0

    normalized_boe_map = (boe_map - np.min(boe_map)) / (np.max(boe_map) - np.min(boe_map))  # convert index to 0-1 range
    boe_colormap = np.uint8((cm.jet_r(normalized_boe_map)[:, :, :3]) * 255)  # convert to colormap, cm: jet
    # boe_colormap[me_map <= config.noise_level] = [255, 255, 255]  # whiten the background pixels

    return boe_map, normalized_boe_map, boe_colormap


def paired_boe(density_sequence_preEVT, density_sequence_postEVT):
    """
    brevity of enhancement
    :param density_sequence_preEVT: a tx1024x1024 DSA series, preEVT
    :param density_sequence_postEVT: a tx1024x1024 DSA series, preEVT
    """

    logger.info("Computing brevity of enhancement colormap")
    min_sequence_len = min(density_sequence_preEVT.shape[0], density_sequence_postEVT.shape[0])
    boe_map_preEVT, me_map_preEVT = _boe(density_sequence_preEVT)
    boe_map_postEVT, me_map_postEVT = _boe(density_sequence_postEVT)

    '''diff'''
    boe_diff = boe_map_postEVT - boe_map_preEVT
    boe_diff[(me_map_preEVT <= config.noise_level) & (me_map_postEVT <= config.noise_level)] = 0
    normalized_boe_diff = boe_diff / np.amax(abs(boe_diff))
    normalized_boe_diff = 0.5 * (1 + normalized_boe_diff)
    boe_diff_colormap = np.uint8((cm.PiYG(normalized_boe_diff)[:, :, :3]) * 255)

    '''preEVT and postEVT colormap'''
    max_boe = min(min_sequence_len - 1, max(np.max(boe_map_preEVT), np.max(boe_map_postEVT)))
    boe_map_preEVT = boe_map_preEVT / max_boe  # convert index to 0-1 range
    boe_colormap_preEVT = np.uint8((cm.jet(boe_map_preEVT)[:, :, :3]) * 255)  # convert to colormap
    boe_colormap_preEVT[me_map_preEVT <= config.noise_level] = [255, 255, 255]
    boe_map_postEVT = boe_map_postEVT / max_boe  # convert index to 0-1 range
    boe_colormap_postEVT = np.uint8((cm.jet(boe_map_postEVT)[:, :, :3]) * 255)  # convert to colormap
    boe_colormap_postEVT[me_map_postEVT <= config.noise_level] = [255, 255, 255]

    label_image = cv.putText(img=np.zeros((100, *boe_colormap_preEVT.shape[1:])), text="BoE", org=(350, 70),
                             fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
    boe_colormap_preEVT = np.concatenate([label_image, boe_colormap_preEVT])
    return boe_colormap_preEVT, boe_colormap_postEVT, boe_diff_colormap


def CBV(img_sequence):
    me_map = np.max(img_sequence, axis=0)
    cbv_map = np.sum(img_sequence, axis=0)

    normalized_cbv_map = (cbv_map - np.min(cbv_map)) / (np.max(cbv_map) - np.min(cbv_map))  # to 0-1 range
    cbv_colormap = np.uint8((cm.jet_r(normalized_cbv_map)[:, :, :3]) * 255)
    # cbv_colormap[me_map <= config.noise_level] = [255, 255, 255]
    return cbv_map, normalized_cbv_map, cbv_colormap


def rCBV(density_sequence_preEVT, density_sequence_postEVT, irf_sequence_preEVT, irf_sequence_postEVT):
    """
    calculates a parametric image based on cerebral blood volume
    :param density_sequence_preEVT: a tx1024x1024 DSA series
    :param density_sequence_postEVT: a tx1024x1024 DSA series
    :param irf_sequence_preEVT: a tx1024x1024 DSA series, preEVT irf sequence
    :param irf_sequence_postEVT: a tx1024x1024 DSA series, postEVT irf sequence
    """

    def _rCBV(density_sequence, irf_sequence):
        me_map = np.max(density_sequence, axis=0)
        rcbv_map = np.sum(irf_sequence, axis=0)

        normalized_rcbv_map = (rcbv_map - np.min(rcbv_map)) / (np.max(rcbv_map) - np.min(rcbv_map))  # to 0-1 range
        rcbv_colormap = np.uint8((cm.jet_r(normalized_rcbv_map)[:, :, :3]) * 255)
        rcbv_colormap[me_map <= config.noise_level] = [255, 255, 255]
        return rcbv_colormap, rcbv_map, me_map

    logger.info("Computing time to peak colormap")
    rcbv_colormap_preEVT, rcbv_preEVT, me_map_preEVT = _rCBV(density_sequence_preEVT, irf_sequence_preEVT)
    rcbv_colormap_postEVT, rcbv_postEVT, me_map_postEVT = _rCBV(density_sequence_postEVT, irf_sequence_postEVT)

    '''diff'''
    rcbv_diff = rcbv_postEVT - rcbv_preEVT
    rcbv_diff[(me_map_preEVT <= config.noise_level) & (me_map_postEVT <= config.noise_level)] = 0
    normalized_rcbv_diff = rcbv_diff / np.amax(abs(rcbv_diff))
    normalized_rcbv_diff = 0.5 * (1 + normalized_rcbv_diff)
    rcbv_diff_colormap = np.uint8((cm.PiYG(normalized_rcbv_diff)[:, :, :3]) * 255)  # convert to colormap, cm: RdYlGn

    label_image = cv.putText(img=np.zeros((100, *rcbv_colormap_preEVT.shape[1:])), text="rCBV", org=(350, 70),
                             fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
    rcbv_colormap_preEVT = np.concatenate([label_image, rcbv_colormap_preEVT])
    return rcbv_colormap_preEVT, rcbv_colormap_postEVT, rcbv_diff_colormap, rcbv_preEVT, rcbv_postEVT


def MTT(density_sequence_preEVT, density_sequence_postEVT, rCBV_preEVT, rCBV_postEVT, rCBF_preEVT, rCBF_postEVT):
    """
    calculates a parametric image based on cerebral blood volume
    :param density_sequence_preEVT: a tx1024x1024 DSA series, preEVT enhancement sequence
    :param density_sequence_postEVT: a tx1024x1024 DSA series, postEVT enhancement sequence
    :param rCBV_preEVT: a 1024x1024 rCBV parameter map
    :param rCBV_postEVT: a 1024x1024 rCBV parameter map
    :param rCBF_preEVT: a 1024x1024 rCBF parameter map
    :param rCBF_postEVT: a 1024x1024 rCBF parameter map
    """
    MTT_preEVT, MTT_postEVT = np.zeros_like(rCBF_preEVT), np.zeros_like(rCBF_postEVT)
    MTT_preEVT[rCBF_preEVT != 0] = rCBV_preEVT[rCBF_preEVT != 0] / rCBF_preEVT[rCBF_preEVT != 0]
    MTT_postEVT[rCBF_postEVT != 0] = rCBV_postEVT[rCBF_postEVT != 0] / rCBF_postEVT[rCBF_postEVT != 0]

    MTT_max = max(np.max(MTT_preEVT), np.max(MTT_postEVT))
    logger.info("Max MTT: {}".format(MTT_max))

    logger.info("Computing time to peak colormap")
    normalized_MTT_preEVT = MTT_preEVT / MTT_max
    mtt_colormap_preEVT = np.uint8((cm.jet_r(normalized_MTT_preEVT)[:, :, :3]) * 255)
    me_map_preEVT = np.max(density_sequence_preEVT, axis=0)
    mtt_colormap_preEVT[me_map_preEVT <= config.noise_level] = [255, 255, 255]

    normalized_MTT_postEVT = MTT_postEVT / MTT_max
    mtt_colormap_postEVT = np.uint8((cm.jet_r(normalized_MTT_postEVT)[:, :, :3]) * 255)
    me_map_postEVT = np.max(density_sequence_postEVT, axis=0)
    mtt_colormap_postEVT[me_map_postEVT <= config.noise_level] = [255, 255, 255]

    '''diff'''
    mtt_diff = MTT_preEVT - MTT_postEVT
    mtt_diff[(me_map_preEVT <= config.noise_level) & (me_map_postEVT <= config.noise_level)] = 0
    normalized_mtt_diff = mtt_diff / np.amax(abs(mtt_diff))
    normalized_mtt_diff = 0.5 * (1 + normalized_mtt_diff)
    mtt_diff_colormap = np.uint8((cm.PiYG(normalized_mtt_diff)[:, :, :3]) * 255)  # convert to colormap, cm: RdYlGn

    label_image = cv.putText(img=np.zeros((100, *mtt_colormap_preEVT.shape[1:])), text="MTT", org=(350, 70),
                             fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
    mtt_colormap_preEVT = np.concatenate([label_image, mtt_colormap_preEVT])
    return mtt_colormap_preEVT, mtt_colormap_postEVT, mtt_diff_colormap


def normalize(img):
    """Outputs image of type unsigned int"""
    image_mean = np.mean(img)
    image_std = np.std(img)
    logger.info('Image mean = {}, image std = {}'.format(image_mean, image_std))
    # img = (img - image_mean) / image_std
    image_minip_norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return image_minip_norm


def remove_non_contrast_frames(preEVT_sequence, postEVT_sequence):
    from phase_classification import phase_prediction
    phase_labels_preEVT = phase_prediction.predict_sequence_phases(preEVT_sequence)
    phase_labels_postEVT = phase_prediction.predict_sequence_phases(postEVT_sequence)
    return np.squeeze(preEVT_sequence[phase_labels_preEVT != 0]), np.squeeze(
        postEVT_sequence[phase_labels_postEVT != 0])


def plot_TIC(sequence_before_filtering, sequence_after_filtering, patient_id, view, prepost):
    """
    Randomly print 10 TICs of the input array
    :param sequence_before_filtering: a tx1024x1024 DSA series
    :param sequence_after_filtering: a tx1024x1024 DSA series
    :param patient_id:
    :param view:
    :param prepost:
    :return:
    """
    fig, axs = plt.subplots(2, 5)
    for r in range(2):
        for c in range(5):
            x, y = np.random.randint(0, 1023), np.random.randint(0, 1023)
            axs[r, c].plot(sequence_before_filtering[:, x, y], color='orange')
            axs[r, c].plot(sequence_after_filtering[:, x, y], color='green')
    plt.savefig(os.path.join(config.current_vis_tic_dirpath, "{}_{}_{}.png".format(patient_id, view, prepost)))
    plt.savefig(os.path.join(config.latest_vis_tic_dirpath, "{}_{}_{}.png".format(patient_id, view, prepost)))
    save_fig(os.path.join(config.current_vis_tic_dirpath, "{}_{}_{}_minip.png".format(patient_id, view, prepost)),
             minip(sequence_before_filtering), minip(sequence_after_filtering))
    save_fig(os.path.join(config.latest_vis_tic_dirpath, "{}_{}_{}_minip.png".format(patient_id, view, prepost)),
             minip(sequence_before_filtering), minip(sequence_after_filtering))
    plt.close()


def extract_AIF(density_sequence, ica_mask, mask_color=(0, 0, 255)):
    """
    :param density_sequence: a tx1024x1024 DSA series, enhancement sequence
    :param ica_mask: 1024x1024 maks image, range 0-1
    :param mask_color: the color for ica mask to show
    """

    AIF = np.zeros((density_sequence.shape[0],))
    for t in range(density_sequence.shape[0]):
        AIF[t] = cv.mean(density_sequence[t], ica_mask.astype(np.uint8))[0]

    vis = minip(255 - density_sequence).astype(np.uint8)
    vis = np.dstack([vis, vis, vis])
    vis[ica_mask == 1] = mask_color
    return AIF, vis


def save_fig_AIF(aif_preEVT, aif_postEVT, vis_AIF_minip_preEVT, vis_AIF_minip_postEVT, patient_id, view):
    t_AIF_preEVT = np.arange(0, config.frame_interval * aif_preEVT.shape[0], config.frame_interval)
    t_AIF_postEVT = np.arange(0, config.frame_interval * aif_postEVT.shape[0], config.frame_interval)
    plt.subplot(221)
    plt.imshow(vis_AIF_minip_preEVT)
    plt.title('preEVT')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(vis_AIF_minip_postEVT)
    plt.title('postEVT')
    plt.axis('off')
    plt.subplot(212)
    plt.plot(t_AIF_preEVT / 1000, aif_preEVT, label="preEVT AIF", color='red')
    plt.plot(t_AIF_postEVT / 1000, aif_postEVT, label="postEVT AIF", color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel('Contrast density [0-255]')
    plt.title('Arterial input functions')
    plt.legend(loc="best")
    plt.savefig(os.path.join(config.current_vis_aif_dirpath, "{}_{}.png".format(patient_id, view)), dpi=1200)
    plt.savefig(os.path.join(config.latest_vis_aif_dirpath, "{}_{}.png".format(patient_id, view)), dpi=1200)
    plt.close()


def compute_2D_DSA(in_serie_path, out_serie_dir, mode='log', registration=False):
    """
    :param registration: Whether to use elastix registration to align frames before subtraction.
    :param in_serie_path: path to un-subtracted angiography
    :param out_serie_dir: output directory of subtracted angiography
    :param mode: how the DSA is computed from angiography,
    either via logorithmic subtraction ('log') or direction subtraction ('direct').
    :return: subtracted serie
    """
    assert mode in ['log', 'direct'], "Invalid subtraction mode"
    serie = dicom_reader.read_file(in_serie_path, defer_size="1 KB", stop_before_pixels=False)
    view = dicom_utils.extract_dicom_view_from_header(serie)
    logger.info("Generating DSA -- mode: {}, Series number: {}, "
                "view: {}, shape: {}".format(mode, serie.SeriesNumber, view, serie.pixel_array.shape))

    img_sequence = serie.pixel_array.astype(float)
    '''Remove black edges to avoid getting -inf for log operation'''
    img_sequence = img_sequence[:, 28:-28, 28:-28]
    serie.Rows, serie.Columns = img_sequence.shape[1], img_sequence.shape[2]

    first_frame = img_sequence[0].copy()
    for i in range(img_sequence.shape[0]):
        # if registration:
        #     mask, _ = elastix.elastix_align_image_pair(sitk.GetImageFromArray(img_sequence[i]),
        #                                                sitk.GetImageFromArray(first_frame),
        #                                                resolution=1, metric=['AdvancedMeanSquares'],
        #                                                n_iteration=['128'])
        if mode == 'log':
            mask, img_sequence[i] = np.log(first_frame), np.log(img_sequence[i])
        else:
            mask = first_frame
        img_sequence[i] = img_sequence[i] - mask

    img_sequence = 255 * (img_sequence - img_sequence.min()) / (img_sequence.max() - img_sequence.min())
    img_sequence = np.array(img_sequence, dtype=np.uint16)
    serie.PixelData = img_sequence.tobytes()

    '''Compute subtraction DSA'''
    out_filepath = os.path.join(out_serie_dir, "{}.dcm".format(serie.SeriesNumber))
    Path(out_filepath).parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(out_filepath, serie, write_like_original=True)

    return serie


def compute_3D_DSA(in_serie_1, in_serie_2, out_serie_dir):
    serie_1 = dicom_reader.read_file(in_serie_1, defer_size="1 KB", stop_before_pixels=False)
    serie_2 = dicom_reader.read_file(in_serie_2, defer_size="1 KB", stop_before_pixels=False)

    # view = dicom_utils.extract_dicom_view_from_header(serie)
    # logger.info("Series number: {}, view: {}, shape: {}".format(serie.SeriesNumber, serie.pixel_array.shape, view))
    #
    # img_sequence = serie.pixel_array.astype(float)
    # '''Remove black edges to avoid getting -inf for log operation'''
    # img_sequence = img_sequence[:, 28:-28, 28:-28]
    # serie.Rows, serie.Columns = img_sequence.shape[1], img_sequence.shape[2]

    img_sequence_1 = serie_1.pixel_array.astype(float)
    img_sequence_2 = serie_2.pixel_array.astype(float)
    # '''Remove black edges to avoid getting -inf for log operation'''
    img_sequence_1 = img_sequence_1[:, 28:-28, 28:-28]
    img_sequence_2 = img_sequence_2[:, 28:-28, 28:-28]
    serie_2.Rows, serie_2.Columns = img_sequence_2.shape[1], img_sequence_2.shape[2]
    assert img_sequence_1.shape[0] == img_sequence_2.shape[0]

    ln_img_sequence_1 = np.log(img_sequence_1)
    ln_img_sequence_2 = np.log(img_sequence_2)

    out_img_sequence = ln_img_sequence_1 - ln_img_sequence_2

    out_img_sequence = 255 * (out_img_sequence - out_img_sequence.min()) / (
            out_img_sequence.max() - out_img_sequence.min())
    out_img_sequence = np.array(out_img_sequence, dtype=np.uint16)
    serie_2.PixelData = out_img_sequence.tobytes()

    '''Compute subtraction DSA'''
    out_filepath = os.path.join(out_serie_dir, "{}.dcm".format(serie_2.SeriesNumber))
    Path(out_filepath).parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(out_filepath, serie_2, write_like_original=True)

    return serie_2

def condensed_main(patid, img_sequence, view, temporal_smoothing=False):
    """

    :param patid: patient id
    :param img_sequence: a numpy array containing a DSA image. The sequence will be normalized
    :param view:
    :return:
    """
    logger.info(f"Generating perfusion visualization --> Patient name: {patid}, View {view}")

    img_sequence = cv.normalize(img_sequence, None, 0, 255, cv.NORM_MINMAX)
    '''Time density curve temporal denoising'''
    # if config.denoising_enabled:
    if temporal_smoothing:
        '''temporal gaussian filtering'''
        img_sequence_filtered = ndimage.gaussian_filter(img_sequence, tuple(args.sigma), mode='nearest')
        '''overwrite variables with the filtered sequences'''
        img_sequence = img_sequence_filtered

    '''MINIP image'''
    img_minip = minip(img_sequence)
    '''Maximum enhancement'''
    me_map, normalized_me_map, me_colormap = me(img_sequence)
    '''Generate contrast density sequence'''
    img_sequence = convert_to_enhancement_sequence(img_sequence)
    toa_colormap, ttp_colormap, wir_colormap = toa_ttp_wir(img_sequence)
    '''Cerebral Blood Volume'''
    cbv_map, normalized_cbv_map, cbv_colormap = CBV(img_sequence)
    '''Brevity of enhancement'''
    boe_map, normalized_boe_map, boe_colormap = boe(img_sequence)

    row_label = cv.rotate(cv.putText(img=np.zeros((100, 1024, 3)), thickness=5, org=(150, 90), fontFace=3,
                                     fontScale=3, text=f"{patid}_{view}",
                                     color=[255, 255, 255]), cv.ROTATE_90_COUNTERCLOCKWISE)
    col_label_1 = cv.putText(img=np.zeros((100, 1124, 3)), text="MinIP", org=(400, 70),
                             fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
    col_label_2 = cv.putText(img=np.zeros((100, 1024, 3)), text="Enhancement", org=(200, 70),
                             fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
    col_label_3 = cv.putText(img=np.zeros((100, 1024, 3)), text="ToA", org=(400, 70),
                             fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
    col_label_4 = cv.putText(img=np.zeros((100, 1024, 3)), text="TTP", org=(400, 70),
                             fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
    col_label_5 = cv.putText(img=np.zeros((100, 1024, 3)), text="WIR", org=(400, 70),
                             fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
    col_label_6 = cv.putText(img=np.zeros((100, 1024, 3)), text="CBV", org=(400, 70),
                             fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
    col_label_7 = cv.putText(img=np.zeros((100, 1024, 3)), text="BoE", org=(400, 70),
                             fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
    vis_col_label = np.concatenate([col_label_1, col_label_2, col_label_3, col_label_4, col_label_5, col_label_6,
                                    col_label_7], axis=1)
    vis = np.concatenate(
        [row_label, np.stack((img_minip,) * 3, axis=-1), me_colormap, toa_colormap, ttp_colormap, wir_colormap,
         cbv_colormap, boe_colormap], axis=1)
    vis = np.concatenate([vis_col_label, vis])
    vis_filepath = os.path.join(args.perf_angio_dir, f"{patid}_{view}.png")
    save_fig(vis_filepath, vis)

def main(args):
    """Extract parametric images from 2D+t DSA series based time information"""

    '''Compute DSA for 2D temporal angiography'''
    # serie_paths = natsorted(glob.glob(os.path.join(config.orig_angio_dir, '**', '*.IMA'), recursive=True))
    # for serie_path in serie_paths:
    #     compute_2D_DSA(serie_path, args.subt_angio_dir, mode=args.dsa_mode)

    '''Compute DSA for 3D rotational angiography'''
    # serie_folders = glob.glob('{}/*/'.format(in_dir), recursive=False)
    # for folder in serie_folders:
    #     logger.info("======== Series path: {} ========".format(folder))
    #     series = sorted(glob.glob(os.path.join(folder, '*.IMA'), recursive=False))
    #     if len(series) != 2:
    #         continue
    #     compute_3D_DSA(series[1], series[0], subtracted_dir)

    subtracted_series = natsorted(glob.glob(os.path.join(args.subt_angio_dir, '**', 'DSA_*.IMA'), recursive=True))
    for subtracted_serie_path in subtracted_series:
        serie = dicom_reader.read_file(subtracted_serie_path, defer_size="1 KB", stop_before_pixels=False)
        view = dicom_utils.extract_dicom_view_from_header(serie)
        logger.info("Generating perfusion visualization --> Smoothing: {}, Patient name: {}, Series number: {}, "
                    "view: {}, shape: {}".format(args.smoothing, serie.PatientName, serie.SeriesNumber, view,
                                                 serie.pixel_array.shape))

        img_sequence = serie.pixel_array.astype(float)
        img_sequence = cv.normalize(img_sequence, None, 0, 255, cv.NORM_MINMAX)

        '''Get annotated ICA masks for pre- and post- EVT'''
        # annotated_ica_mask_filepath = os.path.join(config.ica_mask_dir, patient_id,
        #                                                   preEVT_name + config.mask_file_name_suffix)
        # ica_mask_preEVT = cv.imread(annotated_ica_mask_filepath, cv.IMREAD_GRAYSCALE)  # range: 0-1

        '''Sequence Registration. Register preEVT sequence to postEVT sequence.'''
        # if config.pre_post_sequence_registration_enabled:
        #     latest_registered_preEVT_sequence_path = os.path.join(config.latest_registered_preEVT_dirpath,
        #                                                           "{}_{}_preEVT.npy".format(patient_id, view))
        #     latest_registered_preEVT_mask_path = os.path.join(config.latest_registered_preEVT_dirpath,
        #                                                       "{}_{}_preEVT_mask.npy".format(patient_id, view))
        #     if config.reuse_pre_post_registration_results and os.path.isfile(
        #             latest_registered_preEVT_sequence_path):
        #         logger.info("Reusing existing registered preEVT sequence.")
        #         preEVT_sequence = np.load(latest_registered_preEVT_sequence_path)
        #         ica_mask_preEVT = np.load(latest_registered_preEVT_mask_path)
        #     else:
        #         logger.info("Registering preEVT sequence to postEVT")
        #         preEVT_sequence, ica_mask_preEVT = \
        #             sequence_registration(postEVT_sequence, preEVT_sequence, patient_id, view,
        #                                   moving_mask=ica_mask_preEVT)
        #         np.save(latest_registered_preEVT_sequence_path, preEVT_sequence)
        #         np.save(latest_registered_preEVT_mask_path, ica_mask_preEVT)

        '''Time density curve temporal denoising'''
        # if config.denoising_enabled:
        if args.smoothing:
            '''temporal gaussian filtering'''
            img_sequence_filtered = ndimage.gaussian_filter(img_sequence, tuple(args.sigma), mode='nearest')
            '''plot TIC curves of a couple of randomly selected pixels.'''
            # plot_TIC(img_sequence, img_sequence_filtered, serie.SeriesNumber, view, serie.SeriesNumber)
            '''overwrite variables with the filtered sequences'''
            img_sequence = img_sequence_filtered

        # '''Extract AIF from the annotated ICA masks'''
        # aif_preEVT, vis_AIF_minip_preEVT = extract_AIF(preEVT_sequence, ica_mask_preEVT, mask_color=(255, 0, 0))
        # aif_postEVT, vis_AIF_minip_postEVT = extract_AIF(postEVT_sequence, ica_mask_postEVT, mask_color=(0, 0, 255))
        # save_fig_AIF(aif_preEVT, aif_postEVT, vis_AIF_minip_preEVT, vis_AIF_minip_postEVT, patient_id, view)
        #
        # '''Calculate impulse residue function sequences for pre- and post- EVT'''
        # latest_irf_preEVT_sequence_filepath = os.path.join(
        #     config.latest_irf_sequence_dirpath, "{}_{}_preEVT.npy".format(patient_id, view))
        # latest_irf_postEVT_sequence_filepath = os.path.join(
        #     config.latest_irf_sequence_dirpath, "{}_{}_postEVT.npy".format(patient_id, view))
        # if config.reuse_irf_sequence_results \
        #         and os.path.isfile(latest_irf_preEVT_sequence_filepath) \
        #         and os.path.isfile(latest_irf_postEVT_sequence_filepath):
        #     logger.info("Reusing pre-/post-EVT impulse residue function sequence results")
        #     irf_preEVT_sequence = np.load(latest_irf_preEVT_sequence_filepath)
        #     irf_postEVT_sequence = np.load(latest_irf_postEVT_sequence_filepath)
        # else:
        #     logger.info("Calculate IRF for pre- and post EVT sequences")
        #     # irf_preEVT_sequence = np.apply_along_axis(
        #     #     lambda m: deconvolution.wiener_deconvolution(m, aif_preEVT, lambd=1), axis=0,
        #     #     arr=preEVT_sequence)
        #     # irf_postEVT_sequence = np.apply_along_axis(
        #     #     lambda m: deconvolution.wiener_deconvolution(m, aif_postEVT, lambd=1), axis=0,
        #     #     arr=postEVT_sequence)

        # irf_preEVT_sequence, _ = deconvolution.svd_deconvolve_3D_with_OI(preEVT_sequence, aif_preEVT, oi_threshold=0.08)
        # irf_postEVT_sequence, _ = deconvolution.svd_deconvolve_3D_with_OI(postEVT_sequence, aif_postEVT, oi_threshold=0.08)

        # '''Cut the two sequences into same duration'''
        # min_sequence_len = min(preEVT_sequence.shape[0], postEVT_sequence.shape[0])
        # preEVT_sequence = preEVT_sequence[:min_sequence_len]
        # postEVT_sequence = postEVT_sequence[:min_sequence_len]
        # irf_preEVT_sequence = irf_preEVT_sequence[:min_sequence_len]
        # irf_postEVT_sequence = irf_postEVT_sequence[:min_sequence_len]
        #
        # np.save(latest_irf_preEVT_sequence_filepath, irf_preEVT_sequence)
        # np.save(latest_irf_postEVT_sequence_filepath, irf_postEVT_sequence)

        '''MINIP image'''
        img_minip = minip(img_sequence)
        '''Maximum enhancement'''
        me_map, normalized_me_map, me_colormap = me(img_sequence)
        '''Generate contrast density sequence'''
        img_sequence = convert_to_enhancement_sequence(img_sequence)
        # '''Time of arrive'''
        # toa_map, normalized_toa_map, toa_colormap = toa(img_sequence)
        # '''Time to peak'''
        # ttp_map, normalized_ttp_map, ttp_colormap = ttp(img_sequence)
        toa_colormap, ttp_colormap, wir_colormap = toa_ttp_wir(img_sequence)
        '''Cerebral Blood Volume'''
        cbv_map, normalized_cbv_map, cbv_colormap = CBV(img_sequence)
        '''Wash in rate'''
        # wir_map, normalized_wir_map, wir_colormap = wir(img_sequence)
        '''Brevity of enhancement'''
        boe_map, normalized_boe_map, boe_colormap = boe(img_sequence)

        # '''Tmax'''
        # tmax_colormap, tmax_map, me_map = tmax(img_sequence, irf_sequence)
        # '''Cerebral Blood Flow'''
        # rCBF_map, rCBF_map = rCBF(img_sequence, irf_sequence)
        # '''Cerebral Blood Volume'''
        # rCBV_map, rCBV_map = rCBV(img_sequence, irf_sequence)
        # '''Mean Transition Time'''
        # mtt_preEVT = MTT(img_sequence, rCBV_map, rCBF_map)
        # '''Wash in rate'''
        # wir_preEVT = wir(img_sequence)
        # '''Brevity of enhancement'''
        # boe_preEVT = boe(img_sequence)

        row_label = cv.rotate(cv.putText(img=np.zeros((100, 1024, 3)), thickness=5, org=(150, 90), fontFace=3,
                                         fontScale=3, text=f"{serie.PatientName}_{serie.SeriesNumber}",
                                         color=[255, 255, 255]), cv.ROTATE_90_COUNTERCLOCKWISE)
        col_label_1 = cv.putText(img=np.zeros((100, 1124, 3)), text="MinIP", org=(400, 70),
                                 fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
        col_label_2 = cv.putText(img=np.zeros((100, 1024, 3)), text="Enhancement", org=(200, 70),
                                 fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
        col_label_3 = cv.putText(img=np.zeros((100, 1024, 3)), text="ToA", org=(400, 70),
                                 fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
        col_label_4 = cv.putText(img=np.zeros((100, 1024, 3)), text="TTP", org=(400, 70),
                                 fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
        col_label_5 = cv.putText(img=np.zeros((100, 1024, 3)), text="WIR", org=(400, 70),
                                 fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
        col_label_6 = cv.putText(img=np.zeros((100, 1024, 3)), text="CBV", org=(400, 70),
                                 fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
        col_label_7 = cv.putText(img=np.zeros((100, 1024, 3)), text="BoE", org=(400, 70),
                                 fontFace=3, fontScale=3, color=[255, 255, 255], thickness=5)
        vis_col_label = np.concatenate([col_label_1, col_label_2, col_label_3, col_label_4, col_label_5, col_label_6,
                                        col_label_7], axis=1)
        vis = np.concatenate(
            [row_label, np.stack((img_minip,) * 3, axis=-1), me_colormap, toa_colormap, ttp_colormap, wir_colormap,
             cbv_colormap, boe_colormap], axis=1)
        vis = np.concatenate([vis_col_label, vis])
        vis_filepath = os.path.join(args.perf_angio_dir, f"{serie.PatientName}_{serie.SeriesNumber}.png")
        save_fig(vis_filepath, vis)


def parse_args():
    """
    Argument parser for the main function
    """
    parser = argparse.ArgumentParser(description='DSA perfusion analysis')
    parser.add_argument("--dsa_mode", type=str, help="Choose how the subtraction is done ['log', 'direct']")
    parser.add_argument("--smoothing", action='store_true')
    parser.add_argument("--sigma", type=int, nargs='+', help="sigma used for gaussian filtering, e.g., 3 1 1")

    return parser.parse_args()


if __name__ == '__main__':
    log_filepath = 'log/{}.log'.format(Path(__file__).stem)
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.FileHandler(log_filepath, mode='w'), logging.StreamHandler(sys.stdout)])

    '''Path settings'''
    args = parse_args()
    # subt_angio_dir = os.path.join(config.subt_angio_dir, args.dsa_mode)
    perf_config_str = 'filter_{}'.format('_'.join(map(str, args.sigma))) if args.smoothing else 'nofilter'
    perf_angio_dir = os.path.join(config.perf_angio_dir, "{}_{}".format(args.dsa_mode, perf_config_str))
    # Path(perf_angio_dir).mkdir(parents=True, exist_ok=True)
    # vars(args)['subt_angio_dir'] = subt_angio_dir
    vars(args)['subt_angio_dir'] = config.subt_angio_dir
    vars(args)['perf_angio_dir'] = perf_angio_dir

    main(args)
    print("Done")
