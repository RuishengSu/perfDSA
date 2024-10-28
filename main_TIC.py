import argparse
import glob
import os
import dicom_utils
from non_parametric_deconvolution import modelfree_deconv
from dsa_seq_align import elastix_group_mc
import pipeline_settings as settings
import utils
from pathlib import Path
import logging
import sys
import pandas as pd
import numpy as np
import pydicom
import cv2 as cv
from scipy.stats import spearmanr, mode
from skimage.transform import resize
import pickle
from tensorflow.keras.models import load_model
import ast
import elastix_dsa as reg
from scipy.interpolate import interp1d
import neurite_plot as ne
from scipy.spatial import distance
import matplotlib.pyplot as plt
from ICA_vessel_segmentation import predict_segmentation
import torch
from torchvision import transforms
from ICA_vessel_segmentation.unet import UNet

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
        logger.error("Missing time info: {}".format(src_dcm))
        return None

    return cum_time_vector
    # non_duplicated_frame_indices = np.where(~pd.DataFrame(cum_time_vector).duplicated())
    # cum_time_vector = cum_time_vector[non_duplicated_frame_indices]
    # seq = ds.pixel_array[non_duplicated_frame_indices]
    # cum_time_vector, seq = cum_time_vector[1:], seq[1:]


def temporal_interp(seq, cum_time_vector, target_fps=1):
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


def plot_tics(aif, fps=1, save_path=None):
    x = np.linspace(0, len(aif) / fps, len(aif))

    f, ax = plt.subplots(1, 1)
    ax.plot(x, aif, label='AIF', color='#1D1D1B')
    ax.legend()
    plt.xlabel("Time(s)")
    plt.ylabel("Concentration")
    plt.legend(prop={'size': 18})
    plt.gcf().set_size_inches(15, 4)
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


def perfDSA(fp, out_dir):
    series_id = Path(fp).stem
    patient_id = utils.get_subject_id(fp)

    '''Preprocessing DSA'''
    # get sequence and sequence info
    # seq_info = df_sequence[(df_sequence['patient_id'] == patient_id) & (df_sequence['filename'] == series_id)].iloc[0]
    # seq_info.pixel_spacing = abs(seq_info.pixel_spacing)
    ds = pydicom.read_file(fp, defer_size="1 KB", stop_before_pixels=False, force=True)
    seq = ds.pixel_array
    seq = resize(seq, (seq.shape[0], 1024, 1024), anti_aliasing=False, preserve_range=True)
    # seq, seq_info = resize_to_1024(seq, seq_info)
    assert 2 ** (ds.BitsStored - 1) < ds.pixel_array.max() < 2 ** ds.BitsStored, \
        "Error: bits stored: {}, pixel value max: {}".format(ds.BitsStored, ds.pixel_array.max())
    seq = seq.astype(np.float32) / (2 ** ds.BitsStored - 1)  # convert to value range 0-1
    img_minip = normalize(np.min(seq, axis=0))

    # motion correction
    # if settings.motion_correction_enabled:
    #     seq = elastix_group_mc(seq)
    '''Segmenting ICA using pre-trained U-Net'''
    # load the AI segmentation model to predict the mask
    net = UNet(n_channels=1, n_classes=2, bilinear=True)
    net.load_state_dict(torch.load(configs.ica_model_path))
    AIF_mask = predict_segmentation(net, img_minip)
    # AIF_coords[:, [1, 0]] = AIF_coords[:, [0, 1]]
    # logger.info("AIF coords: [{}-{}, {}-{}]".format(AIF_coords[:,0].min(), AIF_coords[:,0].max(),
    #                                                 AIF_coords[:,1].min(), AIF_coords[:,1].max()))

    '''Preparing DSA for computing perfusion parametric images.'''
    # interpolate sequence to fixed time resolution
    target_fps = 2
    cum_time_vector = get_cum_time_vector(ds)
    seq = temporal_interp(seq, cum_time_vector, target_fps=target_fps)

    seq = seq[1:, :, :]  # exclude the first frame as it is often unsubtracted.
    seq = 1 - seq  # inverse dsa image to have positive TICs when contrasts arrive
    seq = seq - np.median(seq)  # deduct median value to have TIC baseline value to be 0.
    seq[seq < 0] = 0

    # parametric image generation using deconvolution
    # aif = seq[:, AIF_coords[0], AIF_coords[1]]
    '''Extracting AIF'''
    masked_ica_vessel = np.ma.masked_array(seq, mask=~np.repeat(AIF_mask[np.newaxis, ...].astype(bool), seq.shape[0], axis=0))
    aif = masked_ica_vessel.mean(axis=(1, 2))
    # aif = np.mean(seq[:, AIF_mask], axis=1)
    logger.info("AIF {}: {}".format(aif.shape, aif))
    # visualize aif
    save_path = os.path.join(settings.output_dir, "aif.png")
    plot_tics(aif, fps=target_fps, save_path=save_path)

    '''Computing deconvolution-basd parametric images'''
    CBV, CBF, MTT, Tmax = modelfree_deconv(seq, aif, 1000.0 / target_fps, hct=0.45, epsilon=1e-9, dtype=np.float32)
    # non-deconv parameters
    peak = np.max(seq, axis=0)

    # visualize parametric images
    minip = 255 * np.min(1 - seq, axis=0)
    minip = minip.astype(np.uint8)
    minip = cv.cvtColor(minip, cv.COLOR_GRAY2BGR)
    contours, _ = cv.findContours(AIF_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    AIF_contour_points = max(contours, key=cv.contourArea)
    cv.drawContours(minip, AIF_contour_points, -1, (256, 0, 0), thickness=-1)

    save_path = os.path.join(settings.output_dir, "vis.png".format(patient_id, series_id))
    ne.slices([minip, CBV, CBF, MTT, Tmax, peak],
              do_colorbars=True, show=False, save_path=save_path, width=20, dpi=600, grid=(2, 3),
              cmaps=['gray', 'jet', 'jet', 'jet', 'jet', 'jet'],
              # imshow_args=[None, {'vmin': xx, 'vmax': xx}, {'vmin': xx, 'vmax': xx},
              #              {'vmin': xx, 'vmax': xx}, {'vmin': xx, 'vmax': xx}],
              titles=['MinIP', 'CBV', f'CBF', f'MTT', f'Tmax', f"Peak intensity"])

    return minip, CBV, CBF, MTT, Tmax, peak


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
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    '''Path settings'''
    if os.path.isfile(configs.input_dicom_path) and '.dcm' in configs.input_dicom_path:
        param_dict = perfDSA(configs.input_dicom_path, configs.output_dir)
        logger.info("Results: {}".format(param_dict))
    elif os.path.isdir(configs.input_dicom_path):
        df = pd.DataFrame(columns=['patient_id', 'series',
                                   'ICA_CBV', 'ICA_CBF', 'ICA_MTT', 'ICA_Tmax', 'ICA_Peak',
                                   'MCA_CBV', 'MCA_CBF', 'MCA_MTT', 'MCA_Tmax', 'MCA_Peak'])
        # df_selection = pd.read_excel("./221209-tic_perfect_frames.xlsx")
        df_selection = pd.read_csv("./230712-tic_selection_with_venous_2.csv")
        series_ids = df_selection['filename'].unique()
        for series_idx, series_row in df_selection.iterrows():
            # if series_idx < 600:
            #     continue
            if series_idx == 435:
                continue
            # if series_row['patient_id'] != "R2580":4            #     continue
            series_path = os.path.join(settings.clean_dicom_path, series_row['patient_id'],
                                       f"{series_row['filename']}.dcm")
            if not os.path.isfile(series_path):
                series_path = os.path.join(settings.mrclean_dicom_path, series_row['patient_id'],
                                           f"{series_row['filename']}.dcm")
            if not os.path.isfile(series_path):
                series_path = os.path.join(settings.mrclean_part3_dicom_path, series_row['patient_id'],
                                           f"{series_row['filename']}.dcm")

            logger.info("==== {}/{} -- Patient: {}, Series: {}".format(
                series_idx + 1, df_selection.shape[0], utils.get_subject_id(series_path), Path(series_path).stem))
            param_dict = perfDSA(series_path)
            df_row = pd.DataFrame([param_dict])
            df = pd.concat([df, df_row], ignore_index=True)
        df.to_csv(settings.output_csv_path, index=False)
    print("Done")
