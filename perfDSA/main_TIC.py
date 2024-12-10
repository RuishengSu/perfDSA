import argparse
import glob
import logging
import os
import pickle
import sys
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from scipy.interpolate import interp1d
from scipy.stats import mode
from skimage.transform import resize
import tensorflow as tf

import perfDSA.elastix_dsa as reg
import perfDSA.neurite_plot as ne
import perfDSA.pipeline_settings as settings
import perfDSA.utils
from perfDSA.filters import frangi
from perfDSA.non_parametric_deconvolution import modelfree_deconv
from perfDSA.vessel_segmentation import predict_segmentation
from perfDSA.vessel_segmentation.unet import UNet

logger = logging.getLogger(__name__)


def warp_seq(img, transformation_matrix):
    frames, rows, cols = img.shape
    for frame_idx in range(frames):
        img[frame_idx, :, :] = cv.warpAffine(img[frame_idx, :, :], transformation_matrix, (cols, rows))
    return img


def get_cum_time_vector(ds):
    if 'FrameTimeVector' in ds:
        if len(ds.FrameTimeVector) != ds.NumberOfFrames:
            logger.warning("Number of Frames ({}) does not match frame time vector length ({}): {}"
                           "".format(ds.NumberOfFrames, len(ds.FrameTimeVector), ds.FrameTimeVector))
            ds.FrameTimeVector = ds.FrameTimeVector[:ds.NumberOfFrames]
        cum_time_vector = np.cumsum(ds.FrameTimeVector)
    elif 'FrameTime' in ds:
        cum_time_vector = int(ds.FrameTime) * np.array(range(ds.NumberOfFrames))
    else:
        logger.error("Missing time info: {} - {}".format(ds.PatientID, ds.SeriesID))
        return None

    return cum_time_vector


def get_max_fps(cum_time_vector):
    cum_time_vector = cum_time_vector[cum_time_vector >= 100]
    cum_time_vector = np.unique(cum_time_vector)
    if len(cum_time_vector) <= 1:
        return None
    frame_times = cum_time_vector[1:] - cum_time_vector[:-1]
    max_fps = 1000 / frame_times.min()
    return max_fps


def temporal_interp(seq, cum_time_vector, target_fps):
    target_frame_interval = 1000.0 / target_fps  # ms
    interp = interp1d(cum_time_vector, seq, axis=0)
    seq = interp(np.arange(cum_time_vector[0], cum_time_vector[-1], target_frame_interval))
    return seq


def detect_landmarks(img, view):
    """Assuming input image is normalized to 0-1"""
    downsample_factor = img.shape[0] / 256
    img = cv.resize(img, (256, 256))[np.newaxis, :, :, np.newaxis]
    # model = load_model(os.path.join(settings.landmark_detection_dir,
    #                                                 '{}_combined.h5'.format(view)), compile=False)
    model = model_ap if view == 'ap' else model_lateral
    heatmaps = model.predict(img)
    ICA_heatmap = heatmaps[0, :, :, 0, 0]

    print(ICA_heatmap.max())
    ICA_coords = np.unravel_index(ICA_heatmap.argmax(), ICA_heatmap.shape, order='F')
    ICA_coords = tuple(int(i * downsample_factor) for i in ICA_coords)

    logger.info("Predicted location of ICA: {}".format(ICA_coords))
    return ICA_coords


def extract_skull_mask(sequence):
    def diffcount(A):
        B = A.copy()
        B.sort(axis=0)
        C = np.diff(B, axis=0) != 0
        D = C.sum(axis=0) + 1
        return D

    sequence = cv.normalize(sequence.copy(), None, 0, 255,
                            cv.NORM_MINMAX)  # Normalize sequence. Output image type: np.uint8
    # seq = seq.astype(np.uint8)
    img = np.min(sequence, axis=0)

    unique_value_img = diffcount(sequence)
    background_mask = np.zeros_like(unique_value_img, dtype=bool)
    if sequence.shape[0] == 2:
        background_mask = (unique_value_img == 1) & (img != 0)
    elif sequence.shape[0] >= 3:
        background_mask = (unique_value_img <= 2) & (img != 0)
    img[background_mask] = 0

    '''Try to get background intensity value. If not succeed, use 255 as default.'''
    background_intensity = 255  # assuming img intensity range [0, 255]
    if np.count_nonzero(background_mask) != 0:
        background_intensity = np.median(np.min(sequence[:, background_mask], axis=0))
    return img, background_intensity, background_mask


def register_ROIs(seq, atlases, seq_info, ICA_coords=None):
    """Register atlases and sequence"""
    patient_id = seq_info['patient_id']
    view = seq_info['view']
    hemisphere = seq_info['hemisphere']
    lateral_orientation = seq_info['lateral_orientation']

    seq = cv.normalize(seq, None, 0, 255, cv.NORM_MINMAX)  # Normalize sequence. Output image type: np.uint8

    '''Extract skull mask'''
    skull_masked_minip, _, _ = extract_skull_mask(seq)

    '''align atlas to current sequence'''
    registration_pickle_path = os.path.join(settings.registration_dirpath, '{}_{}.pickle'.format(patient_id, view))
    if settings.reuse_registration and os.path.isfile(registration_pickle_path):
        with open(registration_pickle_path, 'rb') as f:
            best_atlas_path, best_transformed_atlas_ICA, best_transformed_atlas_MCA, \
                best_skull_masked_minip_atlas, best_transformed_minip_atlas = pickle.load(f)
        logger.info("Reusing registered atlas for patient {} and view {}: {}".format(patient_id, view, best_atlas_path))
    else:
        logger.info("Finding and registering the best matching atlas to sequence.")

        max_metric_value = 0
        for idx_atlas, atlas in atlases.iterrows():
            atlas_ds = pydicom.read_file(atlas['original'], defer_size="1 KB", stop_before_pixels=False, force=True)
            atlas_ds.pixel_spacing = abs(float(
                atlas_ds.ImagerPixelSpacing[0]) * atlas_ds.DistanceSourceToPatient / atlas_ds.DistanceSourceToDetector)
            atlas_ds.hemisphere = hemisphere
            atlas_ICA = cv.imread(atlas['ICA_mask'], cv.IMREAD_GRAYSCALE)
            atlas_MCA = cv.imread(atlas['MCA_mask'], cv.IMREAD_GRAYSCALE)
            atlas_seq = cv.normalize(atlas_ds.pixel_array, None, 0, 255, cv.NORM_MINMAX)

            atlas_flipped = False
            if (view == 'ap') and (hemisphere not in atlas['original']):
                atlas_seq, atlas_flipped = np.flip(atlas_seq, axis=2), True
                atlas_ICA, atlas_MCA = np.fliplr(atlas_ICA), np.fliplr(atlas_MCA)
            if (view == 'lateral') and lateral_orientation == 'left':
                atlas_seq, atlas_flipped = np.flip(atlas_seq, axis=2), True
                atlas_ICA, atlas_MCA = np.fliplr(atlas_ICA), np.fliplr(atlas_MCA)

            '''pre-registration based on landmarks'''
            if settings.landmark_preregistration_enabled:
                atlas_ICA_coords = list(eval(atlas['ICA_coords']))
                if atlas_flipped:
                    atlas_ICA_coords[0] = 1023 - atlas_ICA_coords[0]
                ICA_translation_matrix = np.float32([[1, 0, ICA_coords[0] - atlas_ICA_coords[0]],
                                                     [0, 1, ICA_coords[1] - atlas_ICA_coords[1]]])
                # if distance.euclidean(atlas_ICA_coords, ICA_coords) < 250:
                atlas_seq = warp_seq(atlas_seq, ICA_translation_matrix)
                atlas_ICA = cv.warpAffine(atlas_ICA, ICA_translation_matrix, (1024, 1024))
                atlas_MCA = cv.warpAffine(atlas_MCA, ICA_translation_matrix, (1024, 1024))

            skull_masked_minip_atlas, _, _ = extract_skull_mask(atlas_seq)

            metric_value, transform_for_atlas = reg.register(skull_masked_minip, skull_masked_minip_atlas)
            logger.info("{}-{}, template: {}, mi: {}".format(patient_id, view, atlas['original'], metric_value))
            if metric_value > max_metric_value:
                max_metric_value = metric_value
                best_atlas_path = atlas['original']
                best_atlas_ICA, best_atlas_MCA = atlas_ICA, atlas_MCA
                best_skull_masked_minip_atlas = skull_masked_minip_atlas
                best_transform_for_atlas = transform_for_atlas
                best_atlas_minip = np.min(atlas_seq, axis=0)
        best_transformed_atlas_ICA = reg.transform_image(best_atlas_ICA, best_transform_for_atlas)
        best_transformed_atlas_ICA = best_transformed_atlas_ICA.astype(np.uint8)
        best_transformed_atlas_MCA = reg.transform_image(best_atlas_MCA, best_transform_for_atlas)
        best_transformed_atlas_MCA = best_transformed_atlas_MCA.astype(np.uint8)
        best_transformed_minip_atlas = reg.transform_image(best_atlas_minip, best_transform_for_atlas)

    '''Backup registration results for reuse'''
    pickle.dump(
        [best_atlas_path, best_transformed_atlas_ICA, best_transformed_atlas_MCA,
         best_skull_masked_minip_atlas, best_transformed_minip_atlas], open(registration_pickle_path, "wb"))
    logger.info("Best atlas for patient {} and view {}: {}".format(patient_id, view, best_atlas_path))

    best_transformed_atlas_ICA[best_transformed_atlas_ICA > 0] = 1
    best_transformed_atlas_MCA[best_transformed_atlas_MCA > 0] = 1
    return best_transformed_atlas_ICA, best_transformed_atlas_MCA


def pad_image(img, to=1024, cval=None):
    pad_h1 = (to - img.shape[0]) // 2
    pad_h2 = to - img.shape[0] - pad_h1
    pad_w1 = (to - img.shape[1]) // 2
    pad_w2 = to - img.shape[1] - pad_w1
    if cval is None:
        cval, _ = mode(img, axis=None, keepdims=False)
    return np.pad(img, ((pad_h1, pad_h2), (pad_w1, pad_w2)), 'constant', constant_values=cval)


def pad_sequence(seq, to=1024):
    out = []
    for i in range(seq.shape[0]):
        out.append(pad_image(seq[i], to=to))
    return np.stack(out, axis=0)


def resize_to_1024(seq, seq_info):
    if seq.shape[1:] != (1024, 1024):
        logger.info("Resizing frames from {} to 1024*1024, "
                    "pixel spacing from {} to {}".format(seq.shape[1:], seq_info['pixel_spacing'],
                                                         seq_info['pixel_spacing'] * (seq.shape[1] / 1024)))
        seq = pad_sequence(seq, to=max(seq.shape[1:]))  # pad frames to square images
        if seq.shape[1] != 1024:
            seq_info['pixel_spacing'] *= (seq.shape[1] / 1024)
            seq = resize(seq, (seq.shape[0], 1024, 1024), anti_aliasing=False, preserve_range=True)
    return seq, seq_info


def resize_to_target_spacing(seq, seq_info, target_spacing=None, masks=None):
    if target_spacing is None:
        target_spacing = abs(seq_info.pixel_spacing) if 0.14 <= abs(seq_info.pixel_spacing) <= 0.16 else 0.15
    if target_spacing != abs(seq_info.pixel_spacing):
        seq_new_size = int(1024 * abs(seq_info.pixel_spacing) / target_spacing)
        seq = resize(seq, (seq.shape[0], seq_new_size, seq_new_size), anti_aliasing=False, preserve_range=True)
        if masks is not None:
            for i, mask in enumerate(masks):
                masks[i] = resize(mask, (seq_new_size, seq_new_size), anti_aliasing=False, preserve_range=True)
        if seq.shape[1] < 1024:
            seq = pad_sequence(seq, to=1024)
            if masks is not None:
                for i, mask in enumerate(masks):
                    masks[i] = pad_image(mask, to=1024, cval=0)

        if seq.shape[1] > 1024:
            crop_size = (seq.shape[1] - 1024) // 2
            seq = seq[:, crop_size:crop_size + 1024, crop_size:crop_size + 1024]
            if masks is not None:
                for i, mask in enumerate(masks):
                    masks[i] = mask[crop_size:crop_size + 1024, crop_size:crop_size + 1024]
        seq_info.pixel_spacing = target_spacing
    if masks is not None:
        if masks is not None:
            for i, mask in enumerate(masks):
                masks[i][mask >= 128] = 255
        return seq, seq_info, masks
    else:
        return seq, seq_info


def plot_tics(aif, tic_ICA, tic_MCA, fps, irf_MCA=None, save_path=None):
    x = np.linspace(0, len(aif) / fps, len(aif))
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    if irf_MCA is not None:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    ax1.plot(x, aif, label='AIF', color='#1D1D1B')
    ax1.legend()
    ax2.plot(x, tic_ICA, label='TIC - ICA', color='tab:red')
    ax2.plot(x, tic_MCA, label='TIC - MCA', color='orange')
    ax2.legend()
    # plt.plot(x, tic_ACA, label='TIC - ACA', color='tab:blue')
    if irf_MCA is not None:
        ax3.plot(x, irf_MCA, label='IRF - MCA', color='orange')
        ax3.legend()

    plt.xlabel("Time(s)")
    # plt.gca().xaxis.set_label_coords(1.01, -.01)
    plt.ylabel("Concentration")
    plt.legend(prop={'size': 18})
    height = (4, 6)[irf_MCA is None]
    plt.gcf().set_size_inches(15, height)
    plt.tight_layout()
    plt.gca().tick_params(color='gray', labelcolor='black')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('gray')
    if save_path is not None:
        plt.savefig(save_path, dpi=1200)
    plt.close()


def find_similar_pixels(img, seed_point, threshold):
    height, width = img.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)
    flags = 4 + (255 << 8) + cv.FLOODFILL_FIXED_RANGE + cv.FLOODFILL_MASK_ONLY
    # flags = 4 | cv.FLOODFILL_FIXED_RANGE | cv.FLOODFILL_MASK_ONLY
    lo_diff, up_diff = (threshold,)*2
    cv.floodFill(img, mask, seed_point, (255,), lo_diff, up_diff, flags)
    mask = mask[1:-1, 1:-1]
    locations = np.argwhere(mask == 255)
    return locations


def normalize(image):
    """
    Normalize the image to [0,255].

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The normalized image.
    """
    # Normalize the image using cv.normalize function
    normalized_image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

    # Convert the normalized image to unsigned int data type
    normalized_image = normalized_image.astype(np.uint8)

    return normalized_image


def truncate(img, img_min=0, img_max=255):
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img


def binarize_image(img, thresh=0):
    # Otsu's thresholding after Gaussian filtering. Inpput image must by of dtype: uint8
    # img = cv.GaussianBlur(img, (5, 5), 0)
    if thresh != 0:
        return cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
    else:
        return cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


def segment(img):
    MIN_VESSEL_INTENSITY = 2
    img_vessel = frangi(img.astype(float), sigmas=(2, 12, 2))
    img_vessel = img_vessel * 255
    img_vessel = truncate(img_vessel, img_min=0, img_max=255)
    _, img_vessel_binary = binarize_image(img_vessel.astype(np.uint8), thresh=settings.FRONGI_INTENSITY_THRES)
    # cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # a boolean array of (width, height) in which False is invalid pixels (vessel) and True is valid pixels (non-vessel)
    vessel_mask = (img_vessel_binary == 255) & (img >= MIN_VESSEL_INTENSITY)
    return vessel_mask


def analyse_TICs(fp):
    series_id = Path(fp).stem
    patient_id = utils.get_mrclean_subject_id(fp)
    # patient_id = utils.get_noiv_subject_id(fp)

    # get sequence and sequence info
    seq_info = df_sequence[(df_sequence['patient_id'] == patient_id) & (df_sequence['filename'] == series_id)].iloc[0]
    # seq_info.pixel_spacing = abs(seq_info.pixel_spacing)
    ds = pydicom.read_file(fp, defer_size="1 KB", stop_before_pixels=False, force=True)
    seq = ds.pixel_array
    seq, seq_info = resize_to_1024(seq, seq_info)
    assert 2 ** (ds.BitsStored - 1) < ds.pixel_array.max() < 2 ** ds.BitsStored, \
        "Error: bits stored: {}, pixel value max: {}".format(ds.BitsStored, ds.pixel_array.max())
    seq = seq.astype(np.float32) / (2 ** ds.BitsStored - 1)  # convert to value range 0-1
    # if settings.respacing_enabled and not np.isnan(seq_info.pixel_spacing):
    #     seq, seq_info = resize_to_target_spacing(seq, seq_info)
    view = seq_info['view']
    img_minip = normalize(np.min(seq, axis=0))

    # motion correction
    # if settings.motion_correction_enabled:
    #     seq = elastix_group_mc(seq)

    '''Registering atlases and sequence'''
    # landmark_pickle_path = os.path.join(settings.landmark_dirpath,
    #                                     '{}_{}.pickle'.format(patient_id, view))
    # if settings.reuse_landmark and os.path.isfile(landmark_pickle_path):
    #     with open(landmark_pickle_path, 'rb') as f:
    #         AIF_coords = pickle.load(f)
    #     logger.info("Reusing detected landmarks for patient {}: {}".format(patient_id, landmark_pickle_path))
    #     logger.info("----> ICA: {}".format(AIF_coords))
    # else:
    #     AIF_coords = detect_landmarks(np.min(seq, axis=0), view)
    #     pickle.dump(AIF_coords, open(landmark_pickle_path, "wb"))
    # 
    # ICA, MCA = register_ROIs(seq, df_atlas[df_atlas['view'] == view], seq_info, ICA_coords=AIF_coords)

    '''Read the manual ICA or MCA mask for the input patient'''
    ICA_mask_paths = glob.glob(os.path.join('./data/MRCLEAN/ica_mca_region_masks_nnunet/', patient_id, '{}*ica*.bmp'.format(seq_info.filename)))
    if ICA_mask_paths:
        ICA_mask_path = ICA_mask_paths[0]
    else:
        logger.warning("ICA mask not found for patient {}, sequence: {}".format(patient_id, seq_info.filename))
        return None
    ICA = np.array(Image.open(ICA_mask_path))
    ICA = np.uint8(resize(ICA, (1024, 1024), anti_aliasing=False, preserve_range=True))
    MCA_mask_paths = glob.glob(os.path.join('./data/MRCLEAN/ica_mca_region_masks_nnunet/', patient_id, '{}*mca*.bmp'.format(seq_info.filename)))
    if MCA_mask_paths:
        MCA_mask_path = MCA_mask_paths[0]
    else:
        logger.warning("MCA mask not found for patient {}, sequence: {}".format(patient_id, seq_info.filename))
        return None
    MCA = np.array(Image.open(MCA_mask_path))
    MCA = np.uint8(resize(MCA, (1024, 1024), anti_aliasing=False, preserve_range=True))

    ''' Try to load manual ICA vessel mask from files'''
    AIF_mask_path = os.path.join('./data/MRCLEAN/manual_ica_vessel_masks/', patient_id, '{}_mask.png'.format(seq_info.filename))
    if not os.path.isfile(AIF_mask_path):
        logger.error("Error: no manual ICA vessel file found! Falling back to ICA vessel prediction.")
        ''' Load the AI segmentation model to predict the ICA vessel mask'''
        net = UNet(n_channels=1, n_classes=2, bilinear=True)
        net.load_state_dict(torch.load(r'vessel_segmentation/checkpoints/ICA_vessel_checkpoint_minip_softmax_size512_rotate.pt'))
        AIF_mask = predict_segmentation(net, img_minip)
        ica_vessel_save_path = os.path.join(settings.output_ica_vessel_dir, patient_id, "{}.png".format(series_id))
        Path(ica_vessel_save_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(AIF_mask*255).save(ica_vessel_save_path)
    else:
        AIF_mask = np.array(Image.open(AIF_mask_path))
        AIF_mask = np.uint8(resize(AIF_mask, (1024, 1024), anti_aliasing=False, preserve_range=True))
    if AIF_mask.max() == 0:
        logger.error("Error: no ICA vessel found in the mask file!")
        return None

    # interpolate sequence to fixed time resolution
    cum_time_vector = get_cum_time_vector(ds)
    target_fps = get_max_fps(cum_time_vector)
    if target_fps is None or target_fps < 0.5 or target_fps > 10:
        logger.error("skipped due to abnormal starting fps {}: {}".format(target_fps, fp))
        return None
    seq = temporal_interp(seq, cum_time_vector, target_fps=target_fps)

    seq = seq[1:, :, :]  # exclude the first frame as it is often unsubtracted.
    seq = 1 - seq  # inverse dsa image to have positive TICs when contrasts arrive
    seq = seq - np.median(seq)  # deduct median value to have TIC baseline value to be 0.
    seq[seq < 0] = 0

    # parametric image generation using deconvolution
    masked_ica_vessel = np.ma.masked_array(seq, mask=~np.repeat(AIF_mask[np.newaxis, ...].astype(bool), seq.shape[0], axis=0))

    aif = masked_ica_vessel.mean(axis=(1, 2))
    CBV, CBF, MTT, Tmax, IRF = modelfree_deconv(seq, aif, 1000.0 / target_fps, hct=0.45, epsilon=1e-9, dtype=np.float32)

    # non-deconv parameters
    peak = np.max(seq, axis=0)

    save_path = os.path.join(settings.output_vis_dir, "{}_{}_CBV.nii".format(patient_id, series_id))
    nib.save(nib.Nifti1Image(np.transpose(CBV)[:, :, np.newaxis], np.eye(4)), save_path)
    save_path = os.path.join(settings.output_vis_dir, "{}_{}_CBF.nii".format(patient_id, series_id))
    nib.save(nib.Nifti1Image(np.transpose(CBF)[:, :, np.newaxis], np.eye(4)), save_path)
    save_path = os.path.join(settings.output_vis_dir, "{}_{}_MTT.nii".format(patient_id, series_id))
    nib.save(nib.Nifti1Image(np.transpose(MTT)[:, :, np.newaxis], np.eye(4)), save_path)
    save_path = os.path.join(settings.output_vis_dir, "{}_{}_Tmax.nii".format(patient_id, series_id))
    nib.save(nib.Nifti1Image(np.transpose(Tmax)[:, :, np.newaxis], np.eye(4)), save_path)
    save_path = os.path.join(settings.output_vis_dir, "{}_{}_peak.nii".format(patient_id, series_id))
    nib.save(nib.Nifti1Image(np.transpose(peak)[:, :, np.newaxis], np.eye(4)), save_path)

    perfused_area = CBV > 0.01

    # gather results - whole area
    ICA_region_mask = ICA.astype(bool)
    ICA_region_pixels = np.sum(ICA_region_mask)
    perfused_ICA_region_mask = ICA_region_mask & perfused_area
    ICA_CBV_whole = np.ma.masked_array(CBV, mask=~perfused_ICA_region_mask).mean()
    ICA_CBF_whole = np.ma.masked_array(CBF, mask=~perfused_ICA_region_mask).mean()
    ICA_MTT_whole = np.ma.masked_array(MTT, mask=~perfused_ICA_region_mask).mean()
    ICA_Tmax_whole = np.ma.masked_array(Tmax, mask=~perfused_ICA_region_mask).mean()
    ICA_peak_whole = np.ma.masked_array(peak, mask=~perfused_ICA_region_mask).mean()

    MCA_region_mask = MCA.astype(bool)
    MCA_region_pixels = np.sum(MCA_region_mask)
    perfused_MCA_region_pixels = MCA_region_mask & perfused_area
    MCA_CBV_whole = np.ma.masked_array(CBV, mask=~perfused_MCA_region_pixels).mean()
    MCA_CBF_whole = np.ma.masked_array(CBF, mask=~perfused_MCA_region_pixels).mean()
    MCA_MTT_whole = np.ma.masked_array(MTT, mask=~perfused_MCA_region_pixels).mean()
    MCA_Tmax_whole = np.ma.masked_array(Tmax, mask=~perfused_MCA_region_pixels).mean()
    MCA_peak_whole = np.ma.masked_array(peak, mask=~perfused_MCA_region_pixels).mean()

    # U-Net based vessel segmentation
    minip = 255 * np.min(1 - seq, axis=0)
    minip = minip.astype(np.uint8)
    net = UNet(n_channels=1, n_classes=2, bilinear=True)
    net.load_state_dict(torch.load(r'vessel_segmentation/checkpoints/vessel_checkpoint_minip_softmax_size1024.pt'))
    vessel_mask = predict_segmentation(net, img_minip)
    vessel_save_path = os.path.join(settings.output_vessel_dir, patient_id, "{}.png".format(series_id))
    Path(vessel_save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vessel_mask*255).save(vessel_save_path)

    # gather results - tissue
    ICA_tissue_mask = ICA.astype(bool) & (~vessel_mask.astype(bool))
    ICA_tissue_pixels = np.sum(ICA_tissue_mask)
    ICA_tissue_percentage = ICA_tissue_pixels/ICA_region_pixels

    perfused_ICA_tissue_mask = ICA_tissue_mask & perfused_area
    perfused_ICA_tissue_pixels = np.sum(perfused_ICA_tissue_mask)
    perfused_ICA_tissue_percentage = perfused_ICA_tissue_mask/ICA_region_pixels

    ICA_CBV_tissue = np.ma.masked_array(CBV, mask=~perfused_ICA_tissue_mask).mean()
    ICA_CBF_tissue = np.ma.masked_array(CBF, mask=~perfused_ICA_tissue_mask).mean()
    ICA_MTT_tissue = np.ma.masked_array(MTT, mask=~perfused_ICA_tissue_mask).mean()
    ICA_Tmax_tissue = np.ma.masked_array(Tmax, mask=~perfused_ICA_tissue_mask).mean()
    ICA_peak_tissue = np.ma.masked_array(peak, mask=~perfused_ICA_tissue_mask).mean()

    MCA_tissue_mask = MCA.astype(bool) & (~vessel_mask.astype(bool))
    MCA_tissue_pixels = np.sum(MCA_tissue_mask)
    MCA_tissue_percentage = MCA_tissue_pixels/MCA_region_pixels

    perfused_MCA_tissue_mask = MCA_tissue_mask & perfused_area
    perfused_MCA_tissue_pixels = np.sum(perfused_MCA_tissue_mask)
    perfused_MCA_tissue_percentage = perfused_MCA_tissue_mask/MCA_region_pixels

    MCA_CBV_tissue = np.ma.masked_array(CBV, mask=~perfused_MCA_tissue_mask).mean()
    MCA_CBF_tissue = np.ma.masked_array(CBF, mask=~perfused_MCA_tissue_mask).mean()
    MCA_MTT_tissue = np.ma.masked_array(MTT, mask=~perfused_MCA_tissue_mask).mean()
    MCA_Tmax_tissue = np.ma.masked_array(Tmax, mask=~perfused_MCA_tissue_mask).mean()
    MCA_peak_tissue = np.ma.masked_array(peak, mask=~perfused_MCA_tissue_mask).mean()

    # gather results - vessel only
    ICA_vessel_mask = ICA.astype(bool) & vessel_mask.astype(bool)
    ICA_vessel_pixels = np.sum(ICA_vessel_mask)
    ICA_vessel_percentage = ICA_vessel_pixels/ICA_region_pixels
    ICA_CBV_vessel = np.ma.masked_array(CBV, mask=~ICA_vessel_mask).mean()
    ICA_CBF_vessel = np.ma.masked_array(CBF, mask=~ICA_vessel_mask).mean()
    ICA_MTT_vessel = np.ma.masked_array(MTT, mask=~ICA_vessel_mask).mean()
    ICA_Tmax_vessel = np.ma.masked_array(Tmax, mask=~ICA_vessel_mask).mean()
    ICA_peak_vessel = np.ma.masked_array(peak, mask=~ICA_vessel_mask).mean()

    MCA_vessel_mask = MCA.astype(bool) & vessel_mask.astype(bool)
    MCA_vessel_pixels = np.sum(MCA_vessel_mask)
    MCA_vessel_percentage = MCA_vessel_pixels/MCA_region_pixels
    MCA_CBV_vessel = np.ma.masked_array(CBV, mask=~MCA_vessel_mask).mean()
    MCA_CBF_vessel = np.ma.masked_array(CBF, mask=~MCA_vessel_mask).mean()
    MCA_MTT_vessel = np.ma.masked_array(MTT, mask=~MCA_vessel_mask).mean()
    MCA_Tmax_vessel = np.ma.masked_array(Tmax, mask=~MCA_vessel_mask).mean()
    MCA_peak_vessel = np.ma.masked_array(peak, mask=~MCA_vessel_mask).mean()

    # visualize average tics
    tic_ICA = np.ma.masked_where(np.tile(ICA == 0, [seq.shape[0], 1, 1]), seq).mean(axis=(1, 2))
    # logger.info("tic_ICA: {}".format(tic_ICA))
    tic_MCA = np.ma.masked_where(np.tile(MCA == 0, [seq.shape[0], 1, 1]), seq).mean(axis=(1, 2))
    # logger.info("tic_MCA: {}".format(tic_MCA))
    irf_MCA = np.ma.masked_where(np.tile(MCA == 0, [seq.shape[0], 1, 1]), IRF).mean(axis=(1, 2))
    # logger.info("irf_MCA: {}".format(irf_MCA))
    save_path = os.path.join(settings.output_tics_dir, "{}_{}.png".format(patient_id, series_id))
    plot_tics(aif, tic_ICA, tic_MCA, target_fps, irf_MCA=irf_MCA, save_path=save_path)

    # visualize parametric images
    minip = cv.cvtColor(minip, cv.COLOR_GRAY2BGR)
    contours, _ = cv.findContours(ICA, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    ICA_contour_points = max(contours, key=cv.contourArea)
    cv.drawContours(minip, ICA_contour_points, -1, (256, 128, 0), thickness=12)
    contours, _ = cv.findContours(MCA, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    MCA_contour_points = max(contours, key=cv.contourArea)
    cv.drawContours(minip, MCA_contour_points, -1, (128, 128, 0), thickness=6)

    # highlight segmented vessels
    # contours, _ = cv.findContours(ICA_vessel_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # ICA_vessel_contour_points = max(contours, key=cv.contourArea)
    # cv.drawContours(minip, ICA_vessel_contour_points, -1, (256, 128, 0), thickness=-1)
    # contours, _ = cv.findContours(MCA_vessel_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # MCA_vessel_contour_points = max(contours, key=cv.contourArea)
    # cv.drawContours(minip, MCA_vessel_contour_points, -1, (256, 128, 0), thickness=-1)

    contours, _ = cv.findContours(AIF_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    AIF_contour_points = max(contours, key=cv.contourArea)
    cv.drawContours(minip, AIF_contour_points, -1, (256, 0, 0), thickness=-1)
    # for AIF_loc in AIF_coords:
    #     cv.circle(minip, AIF_loc, radius=1, color=(256, 0, 0), thickness=-1)

    save_path = os.path.join(settings.output_vis_dir, "{}_{}.png".format(patient_id, series_id))
    ne.slices([minip, CBV, CBF, MTT, Tmax, peak],
              do_colorbars=True, show=False, save_path=save_path, width=20, dpi=600, grid=(2, 3),
              cmaps=['gray', 'jet', 'jet', 'jet', 'jet', 'jet'],
              # imshow_args=[None, {'vmin': xx, 'vmax': xx}, {'vmin': xx, 'vmax': xx},
              #              {'vmin': xx, 'vmax': xx}, {'vmin': xx, 'vmax': xx}],
              titles=['MinIP', f'CBV(ICA={ICA_CBV_whole:.2f}, MCA={MCA_CBV_whole:.2f})',
                      f'CBF(ICA={ICA_CBF_whole:.2f}, MCA={MCA_CBF_whole:.2f})',
                      f'MTT(ICA={ICA_MTT_whole:.2f}, MCA={MCA_MTT_whole:.2f})',
                      f'Tmax(ICA={ICA_Tmax_whole:.2f}, MCA={MCA_Tmax_whole:.2f})',
                      f"Peak intensity(ICA={ICA_peak_whole:.2f}, MCA={MCA_peak_whole:.2f})"])

    return {'patient_id': patient_id, 'series': series_id,
            'ICA_region_pixels': ICA_region_pixels, 'ICA_CBV_whole': ICA_CBV_whole, 'ICA_CBF_whole': ICA_CBF_whole,
            'ICA_MTT_whole': ICA_MTT_whole, 'ICA_Tmax_whole': ICA_Tmax_whole, 'ICA_peak_whole': ICA_peak_whole,

            'MCA_region_pixels': MCA_region_pixels, 'MCA_CBV_whole': MCA_CBV_whole, 'MCA_CBF_whole': MCA_CBF_whole,
            'MCA_MTT_whole': MCA_MTT_whole, 'MCA_Tmax_whole': MCA_Tmax_whole, 'MCA_peak_whole': MCA_peak_whole,

            'ICA_tissue_pixels': ICA_tissue_pixels, 'ICA_tissue_percentage': ICA_tissue_percentage,
            'perfused_ICA_tissue_pixels': perfused_ICA_tissue_pixels,
            'perfused_ICA_tissue_percentage': perfused_ICA_tissue_percentage,
            'ICA_CBV_tissue': ICA_CBV_tissue, 'ICA_CBF_tissue': ICA_CBF_tissue, 'ICA_MTT_tissue': ICA_MTT_tissue,
            'ICA_Tmax_tissue': ICA_Tmax_tissue, 'ICA_peak_tissue': ICA_peak_tissue,

            'MCA_tissue_pixels': MCA_tissue_pixels, 'MCA_tissue_percentage': MCA_tissue_percentage,
            'perfused_MCA_tissue_pixels': perfused_MCA_tissue_pixels,
            'perfused_MCA_tissue_percentage': perfused_MCA_tissue_percentage,
            'MCA_CBV_tissue': MCA_CBV_tissue, 'MCA_CBF_tissue': MCA_CBF_tissue, 'MCA_MTT_tissue': MCA_MTT_tissue,
            'MCA_Tmax_tissue': MCA_Tmax_tissue, 'MCA_peak_tissue': MCA_peak_tissue,

            'ICA_vessel_pixels': ICA_vessel_pixels, 'ICA_vessel_percentage': ICA_vessel_percentage,
            'ICA_CBV_vessel': ICA_CBV_vessel, 'ICA_CBF_vessel': ICA_CBF_vessel, 'ICA_MTT_vessel': ICA_MTT_vessel,
            'ICA_Tmax_vessel': ICA_Tmax_vessel, 'ICA_peak_vessel': ICA_peak_vessel,

            'MCA_vessel_pixels': MCA_vessel_pixels, 'MCA_vessel_percentage': MCA_vessel_percentage,
            'MCA_CBV_vessel': MCA_CBV_vessel, 'MCA_CBF_vessel': MCA_CBF_vessel, 'MCA_MTT_vessel': MCA_MTT_vessel,
            'MCA_Tmax_vessel': MCA_Tmax_vessel, 'MCA_peak_vessel': MCA_peak_vessel,
            }


def parse_args():
    """
    Argument parser for the main function
    """
    parser = argparse.ArgumentParser(description='DSA perfusion analysis')
    parser.add_argument("dicom-path", type=str, help="Input dicom file or dir path")
    # parser.add_argument("--smoothing", action='store_true')
    # parser.add_argument("--sigma", type=int, nargs='+', help="sigma used for gaussian filtering, e.g., 3 1 1")

    return parser.parse_args()


if __name__ == '__main__':
    log_filepath = 'log/{}.log'.format(Path(__file__).stem)
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.FileHandler(log_filepath, mode='w'), logging.StreamHandler(sys.stdout)])

    # Global variables
    df_sequence = pd.read_csv("./240715-tic_selection_with_venous_2.csv")
    model_ap = tf.keras.models.load_model(os.path.join(settings.landmark_detection_dir, 'ap_combined.h5'), compile=False)
    model_lateral = tf.keras.models.load_model(os.path.join(settings.landmark_detection_dir, 'lateral_combined.h5'), compile=False)

    """Detect ICA and M1 coordinates on atlases with Vincent's method"""
    if settings.reuse_landmark and os.path.isfile(settings.df_atlas_path):
        df_atlas = pd.read_csv(settings.df_atlas_path)
    else:
        df_atlas = pd.DataFrame(columns=['original', 'view', 'ICA_mask', 'MCA_mask', 'ICA_coords'])
        atlases = [[f, f.replace('original', 'masks_ICA') + '.bmp', f.replace('original', 'masks_MCA') + '.bmp']
                   for f in glob.glob(os.path.join(settings.atlas_dir, 'original', '*'))]
        for atlas_path in atlases:
            view = 'ap' if 'ap' in atlas_path[0] else 'lateral'
            atlas = pydicom.read_file(atlas_path[0], defer_size="1 KB", stop_before_pixels=False, force=True)
            atlas = cv.normalize(atlas.pixel_array, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            atlas = np.min(atlas, axis=0)
            ICA_coords = detect_landmarks(atlas, view)
            df_row = pd.DataFrame([{'original': atlas_path[0], 'view': view, 'ICA_mask': atlas_path[1],
                                    'MCA_mask': atlas_path[2], 'ICA_coords': ICA_coords}])
            df_atlas = df_atlas.append([df_atlas, df_row], ignore_index=True)
        df_atlas.to_csv(settings.df_atlas_path, index=False)

    '''Path settings'''
    # args = parse_args()
    if os.path.isfile(settings.clean_dicom_path) and '.dcm' in settings.clean_dicom_path:
        param_dict = analyse_TICs(settings.clean_dicom_path)
        logger.info("Results: {}".format(param_dict))
    elif os.path.isdir(settings.clean_dicom_path):
        df = pd.DataFrame(columns=['patient_id', 'series'])
        # df_selection = pd.read_excel("./221209-tic_perfect_frames.xlsx")
        df_selection = pd.read_csv("./240715-tic_selection_with_venous_2.csv")
        # df_selection = pd.read_csv("./240418-noiv_selection.csv")

        # matthijs_selection = pd.read_csv("./240715-tic_selection_restructure_dsa2.csv").set_index('patient_id')
        series_ids = df_selection['filename'].unique()
        for series_idx, series_row in df_selection.iterrows():
            # if series_idx > 1:
            #     continue
            # if series_row['patient_id'] not in matthijs_selection.index:
            #     logger.warning("{}: Skipped as it is not in Matthijs' selection!".format(series_row['patient_id']))
            #     continue
            if series_idx == 1765:
                continue
            # if series_idx != 1853:
            #     continue
            # if series_row['filename'] != "SN18_Vlateral_SOP1.3.6.1.4.1.40744.9.281725342013688859316614335653335960889":
            #     continue
            # if series_row['patient_id'] != "R0825":
            #     continue
            series_path = os.path.join(settings.clean_dicom_path, series_row['patient_id'],
                                       f"{series_row['filename']}.dcm")
            if not os.path.isfile(series_path):
                series_path = os.path.join(settings.mrclean_dicom_path, series_row['patient_id'],
                                           f"{series_row['filename']}.dcm")
            if not os.path.isfile(series_path):
                series_path = os.path.join(settings.mrclean_part3_dicom_path, series_row['patient_id'],
                                           f"{series_row['filename']}.dcm")

            logger.info("==== {}/{} -- Patient: {}, Series: {}".format(
                series_idx + 1, df_selection.shape[0], utils.get_mrclean_subject_id(series_path), Path(series_path).stem))
            if not os.path.isfile(series_path):
                logger.warning("Skipped as not such file or directory found: {}".format(series_path))
                continue
            param_dict = analyse_TICs(series_path)
            if param_dict is None:
                continue
            df_row = pd.DataFrame([param_dict])
            df = pd.concat([df, df_row], ignore_index=True)
        df.to_csv(settings.output_csv_path, index=False)
    print("Done")
