import argparse
import logging
import sys
from pathlib import Path
import os
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
import pydicom
from unet import UNet, TemporalUNet, ConvLSTM, ConvGRU
from glob import glob
import pandas as pd
from scipy.interpolate import interp1d
import torch.nn.functional as F

def cut_seq(seq, max_len):
    if seq.shape[0] > max_len:
        if np.sum(seq[0, ...]) >= np.sum(seq[-1, ...]):
            seq = seq[1:]
        else:
            seq = seq[:-1]
        seq = cut_seq(seq, max_len=max_len)
    return seq


def load_and_preprocess_dicom(img_path):
    ds = pydicom.dcmread(img_path, defer_size="1 KB", stop_before_pixels=False, force=True)
    assert 2 ** (ds.BitsStored - 1) < ds.pixel_array.max() < 2 ** ds.BitsStored, \
        "Error: bits stored: {}, pixel value max: {}".format(ds.BitsStored, ds.pixel_array.max())

    cum_time_vector = None
    if ('FrameTimeVector' in ds) and (ds.FrameTimeVector is not None):
        if len(ds.FrameTimeVector) != ds.NumberOfFrames:
            print("Warning: number of Frames ({}) does not match frame time vector length ({}): {}"
                  "".format(ds.NumberOfFrames, len(ds.FrameTimeVector), ds.FrameTimeVector))
            ds.FrameTimeVector = ds.FrameTimeVector[:ds.NumberOfFrames]
        cum_time_vector = np.cumsum(ds.FrameTimeVector)
    elif 'FrameTime' in ds:
        cum_time_vector = int(ds.FrameTime) * np.array(range(ds.NumberOfFrames))
    else:
        print("Error: missing time info: {}".format(img_path))

    seq = ds.pixel_array
    if cum_time_vector is not None:
        non_duplicated_frame_indices = np.where(~pd.DataFrame(cum_time_vector).duplicated())
        cum_time_vector = cum_time_vector[non_duplicated_frame_indices]
        seq = ds.pixel_array[non_duplicated_frame_indices]
        # cum_time_vector = [e for i, e in enumerate(cum_time_vector) if i not in duplicated_frame_indices]
        # remove the first frame as it is most likely a non-contrast frame or an un-subtracted frame
        cum_time_vector, seq = cum_time_vector[1:], seq[1:]

        desired_frame_interval = 250  # ms
        interp = interp1d(cum_time_vector, seq, axis=0)
        seq = interp(np.arange(cum_time_vector[0], cum_time_vector[-1], desired_frame_interval))

    MAX_LEN = 20  # Shorten unnecessarily long sequences.
    if seq.shape[0] > MAX_LEN:
        print("Warning: sequence is unnecessarily long ({}), "
              "cutting it to {} frames based on minimum contrast.".format(seq.shape[0], MAX_LEN))
    seq = cut_seq(seq, max_len=MAX_LEN)

    seq = np.transpose(255 * (seq.astype(np.float32) / (2 ** ds.BitsStored - 1)), (1, 2, 0))

    return seq


def load_image(img_path, img_size):
    if '.nii' in img_path:
        img_obj = nib.load(img_path)
        img = np.transpose(img_obj.get_fdata(), (1, 0, 2))
    elif '.dcm' in img_path:
        img = load_and_preprocess_dicom(img_path)
    else:
        img = np.asarray(Image.open(img_path))

    newW, newH = img_size, img_size
    assert newW > 0 and newH > 0, 'Image size is too small, resized images would have no pixel'

    if img.ndim == 2:
        img = img[np.newaxis, ...]
        img = resize(img, (img.shape[0], newW, newH), anti_aliasing=False, preserve_range=True)
    else:
        img = np.transpose(img, (2, 0, 1))
        img = resize(img, (img.shape[0], newW, newH), anti_aliasing=False, preserve_range=True)
        img = img[:, np.newaxis, ...]
    img = img / 255

    return img


def predict(net, img, out_img_path=None, device='cuda'):
    net.eval()
    img = torch.as_tensor(img.copy()).float().contiguous().to(device=device, dtype=torch.float32)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        masks_pred = net(img)
        # masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
        mask_pred = Image.fromarray((F.sigmoid(masks_pred) > 0.5).cpu().detach().numpy().astype(np.uint8))
    if out_img_path is not None:
        Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)
        mask_pred.save(out_img_path)
    return mask_pred


def perfDSA(in_img_path, out_img_path, model_path="./models/best_model_ica_top.pt", device='cuda'):
    """PerfDSA"""

    '''Load the perfDSA segmentation model'''
    model = torch.load(model_path, map_location=device)
    logging.info(f'Model loaded from {model_path}')

    '''Segmentation'''
    if os.path.isfile(in_img_path):
        seq = load_image(in_img_path, img_size = 1024)
    else:
        ValueError("Input file not found: {}".format(in_img_path))

    ica_top_seg_mask = predict(model, in_img, out_img_path)



def get_args():
    parser = argparse.ArgumentParser(description='perfDSA to compute perfusion cerebral DSA')
    parser.add_argument('in_img', '-i', help='Input image to be segmented.')
    parser.add_argument('out_img', '-o', default='./out.png', help='Segmentation result image.')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    '''Global settings'''
    args = get_args()
    perfDSA(args.in_img_path, args.out_img_path)
    logging.info("Done!")
