import glob
import logging
import os
import random
import sys
from pathlib import Path

import cv2 as cv
import imageio
import nibabel as nib
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.interpolate import interp1d
import pydicom
from skimage.transform import resize
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def normalize(img):
    """Outputs image of type unsigned int"""
    image_minip_norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    return image_minip_norm.astype(np.uint8)


def cut_seq(seq, max_len):
    if seq.shape[0] > max_len:
        if np.sum(seq[0, ...]) >= np.sum(seq[-1, ...]):
            seq = seq[1:]
        else:
            seq = seq[:-1]
        seq = cut_seq(seq, max_len=max_len)
    return seq


def prepare_minip(row, mode):  # mode = 'train', 'val', or 'test'
    """Preparing minip for model input."""
    dcm_path = os.path.join(all_data_dir, "DSA", row.patient_id, "{}.dcm".format(row.filename))
    ds = pydicom.dcmread(dcm_path, defer_size="1 KB", stop_before_pixels=False, force=False)
    assert 2 ** (ds.BitsStored - 1) < ds.pixel_array.max() < 2 ** ds.BitsStored, \
        "Error: bits stored: {}, pixel value max: {}".format(ds.BitsStored, ds.pixel_array.max())
    seq = ds.pixel_array
    if seq.shape[1:] != (1024, 1024):
        seq = resize(seq, (seq.shape[0], 1024, 1024), anti_aliasing=False, preserve_range=True)
    seq = 255 * (seq.astype(np.float32) / (2 ** ds.BitsStored - 1))
    img_minip = np.min(seq, axis=0).astype(np.uint8)
    dst_minip_path = "./data/ICATopSeg/{}/minip/{}.png".format(mode, row.filename)
    Path(dst_minip_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving minip to {}".format(dst_minip_path))
    imageio.imwrite(dst_minip_path, img_minip)
    # plt.imsave(dst_minip_path, img_minip, cmap=cm.gray)


def prepare_sequence_and_minip(row, mode):  # mode = 'train', 'val', or 'test'
    """Preparing nifti and minip for model input."""

    '''1. Converting raw dicom to nifti sequences with fixed 1 fps'''
    patient_id = row.patient_id
    dst_nii_path = os.path.join(input_files.prepared_data_out_dir, mode, 'imgs_sequence',
                                '{}_{}'.format(patient_id, "{}.nii".format(row.filename)))
    Path(dst_nii_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Convert DSA to 1 fps and save to {}".format(dst_nii_path))

    src_dcm = os.path.join(input_files.raw_dicom_dir, patient_id, "{}.dcm".format(row.filename))
    ds = dicom_reader.read_file(src_dcm, defer_size="1 KB", stop_before_pixels=False, force=True)
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
        return
    non_duplicated_frame_indices = np.where(~pd.DataFrame(cum_time_vector).duplicated())
    cum_time_vector = cum_time_vector[non_duplicated_frame_indices]
    seq = ds.pixel_array[non_duplicated_frame_indices]
    # cum_time_vector = [e for i, e in enumerate(cum_time_vector) if i not in duplicated_frame_indices]
    # remove the first frame as it is most likely a non-contrast frame or an un-subtracted frame
    cum_time_vector, seq = cum_time_vector[1:], seq[1:]

    desired_frame_interval = 1000  # ms
    if patient_id != "R0365":  # The frame time info in dicom header of this patient is wrong.
        interp = interp1d(cum_time_vector, seq, axis=0)
        seq = interp(np.arange(cum_time_vector[0], cum_time_vector[-1], desired_frame_interval))

    MAX_LEN = 20  # Shorten unnecessarily long sequences.
    if seq.shape[0] > MAX_LEN:
        logger.warning("Sequence is unnecessarily long, "
                       "cutting it to {} frames based on minimum contrast.".format(MAX_LEN))
    seq = cut_seq(seq, max_len=MAX_LEN)

    seq = normalize(seq)
    ds.NumberOfFrames = seq.shape[0]
    ds.FrameTimeVector = list(desired_frame_interval * np.array(range(seq.shape[0])))
    ds.BitsAllocated = 8
    ds.PixelData = seq.tobytes()
    # pydicom.dcmwrite(dst_dcm_path, ds, write_like_original=False)
    seq = seq.transpose((2, 1, 0))
    nii_image = nib.Nifti1Image(seq, np.eye(4))
    nib.save(nii_image, dst_nii_path)

    seq = seq.transpose((2, 1, 0))
    '''2. Preparing MinIP images'''
    img_minip = np.min(seq, axis=0)
    img_minip = normalize(img_minip)
    dst_minip_path = os.path.join(input_files.prepared_data_out_dir, mode, 'imgs_minip',
                                  '{}.png'.format(Path(dst_nii_path).stem))
    Path(dst_minip_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving minip to {}".format(dst_minip_path))
    imageio.imwrite(dst_minip_path, img_minip)
    # plt.imsave(dst_minip_path, img_minip, cmap=cm.gray)
    return len(cum_time_vector)


def prepare_masks(row, mode):  # mode = 'train', 'val', or 'test'
    """Prepare artery-vein segmentation ground-truth segmentations"""
    patient_id = row.patient_id

    '''1. Preparing ICA top vessel mask'''
    ica_top_vessel_mask_path = os.path.join(all_data_dir, "masks", patient_id, "{}_mask.png".format(row.filename))
    # ica_vessel_mask_path = "./data/ICATopSeg/all/masks/{}_{}-mask-ica.bmp".format(patient_id, row.filename)
    assert os.path.isfile(ica_top_vessel_mask_path)
    ica_top_vessel_mask = cv.imread(ica_top_vessel_mask_path, cv.IMREAD_GRAYSCALE)
    if ica_top_vessel_mask != (1024, 1024):
        ica_top_vessel_mask = resize(ica_top_vessel_mask, (1024, 1024), anti_aliasing=False, preserve_range=True)
    ica_top_vessel_mask = np.asarray(ica_top_vessel_mask >= 128, dtype=np.uint8)
    dst_mask_path = "./data/ICATopSeg/{}/masks/{}.png".format(mode, row.filename)
    Path(dst_mask_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving mask to {}".format(dst_mask_path))
    imageio.imwrite(dst_mask_path, ica_top_vessel_mask)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    all_data_dir = "./data/ICATopSeg/all"
    all_csv_path = "./data/ICATopSeg/all.xlsx"

    # df_info = pd.read_excel(all_csv_path)
    # df_info = df_info[df_info['AIF']]
    # df_patients = df_info[['patient_id', 'mrs_rev_di']].drop_duplicates()
    # train_patients, temp_patients = train_test_split(df_patients, test_size=0.5, stratify=df_patients['mrs_rev_di'], random_state=42)
    # val_patients, test_patients = train_test_split(temp_patients, test_size=0.6, stratify=temp_patients['mrs_rev_di'], random_state=42)
    #
    # df_test = df_info[df_info['patient_id'].isin(test_patients['patient_id'])]
    # df_train = df_info[df_info['patient_id'].isin(train_patients['patient_id'])]
    # df_val = df_info[df_info['patient_id'].isin(val_patients['patient_id'])]
    # logger.info("Total patients: {}".format(df_info['patient_id'].nunique()))
    # logger.info("Training patients: {}".format(df_train['patient_id'].nunique()))
    # logger.info("Validation patients: {}".format(df_val['patient_id'].nunique()))
    # logger.info("Testing patients: {}".format(df_test['patient_id'].nunique()))
    # df_info.to_csv("./data/ICATopSeg/all.csv", index=False)
    # df_test.to_csv("./data/ICATopSeg/test.csv", index=False)
    # df_train.to_csv("./data/ICATopSeg/train.csv", index=False)
    # df_val.to_csv("./data/ICATopSeg/val.csv", index=False)

    df_all = pd.read_csv("./data/ICATopSeg/all.csv")
    df_train = pd.read_csv("./data/ICATopSeg/train.csv")
    df_val = pd.read_csv("./data/ICATopSeg/val.csv")
    df_test = pd.read_csv("./data/ICATopSeg/test.csv")
    print("Total set - patients: {}, images: {}".format(df_all['patient_id'].nunique(), df_all.shape[0]))
    print("Training set - patients: {}, images: {}".format(df_train['patient_id'].nunique(), df_train.shape[0]))
    print("Validation set - patients: {}, images: {}".format(df_val['patient_id'].nunique(), df_val.shape[0]))
    print("Test set - patients: {}, images: {}".format(df_test['patient_id'].nunique(), df_test.shape[0]))

    '''Prepare sequences'''
    for idx, row in enumerate(df_train.itertuples()):
        logger.info("Preparing training set -- {}/{}, patient: {}".format(idx+1, len(df_train), row.patient_id))
        prepare_minip(row, mode='train')
        prepare_masks(row, mode='train')
    for idx, row in enumerate(df_val.itertuples()):
        logger.info("Preparing validation set -- {}/{}, patient: {}".format(idx+1, len(df_val), row.patient_id))
        prepare_minip(row, mode='val')
        prepare_masks(row, mode='val')
    for idx, row in enumerate(df_test.itertuples()):
        logger.info("Preparing test set -- {}/{}, patient: {}".format(idx+1, len(df_test), row.patient_id))
        prepare_minip(row, mode='test')
        prepare_masks(row, mode='test')

    logger.info("done!")
