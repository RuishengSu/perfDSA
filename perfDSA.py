import argparse
import logging
import sys
import numpy as np
from matplotlib import pyplot as plt

import nibabel as nib
from PIL import Image
from skimage.transform import resize
import pydicom
import pandas as pd
from scipy.interpolate import interp1d
import cv2 as cv

from DSASequence import DSASequence

def cut_seq(seq, max_len):
    if seq.shape[0] > max_len:
        if np.sum(seq[0, ...]) >= np.sum(seq[-1, ...]):
            seq = seq[1:]
        else:
            seq = seq[:-1]
        seq = cut_seq(seq, max_len=max_len)
    return seq


def load_and_preprocess_dicom(img_path, desired_frame_interval = 250):
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

        interp = interp1d(cum_time_vector, seq, axis=0)
        seq = interp(np.arange(cum_time_vector[0], cum_time_vector[-1], desired_frame_interval))

    # MAX_LEN = 20  # Shorten unnecessarily long sequences.
    # if seq.shape[0] > MAX_LEN:
    #     print("Warning: sequence is unnecessarily long ({}), "
    #           "cutting it to {} frames based on minimum contrast.".format(seq.shape[0], MAX_LEN))
    # seq = cut_seq(seq, max_len=MAX_LEN)

    seq = np.transpose(255 * (seq.astype(np.float32) / (2 ** ds.BitsStored - 1)), (1, 2, 0))

    return seq


def load_image(img_path, img_size, desired_frame_interval = 250):
    if '.nii' in img_path:
        img_obj = nib.load(img_path)
        img = np.transpose(img_obj.get_fdata(), (1, 0, 2))
    elif '.dcm' in img_path:
        img = load_and_preprocess_dicom(img_path, desired_frame_interval=desired_frame_interval)
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


def get_args():
    parser = argparse.ArgumentParser(description='perfDSA to compute perfusion cerebral DSA')
    parser.add_argument('-i', dest='in_dcm', help='Input dicom file to be processed.')
    parser.add_argument('-o', dest='out_img', default='./out.png', help='Output image path.')
    parser.add_argument('-m', dest='model_path', default="./models/best_model_ica_top.pt", help='Path to the ICA top segmentation model.')
    parser.add_argument('-d', dest='device', default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu").')
    parser.add_argument('-f', dest='desired_frame_interval', default=250, type=int, help='Desired frame interval in ms for the uniformly-timed sequence.')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    '''Global settings'''
    args = get_args()

    '''Load the dicom file as a numpy array'''
    frames = load_and_preprocess_dicom(args.in_dcm, desired_frame_interval=args.desired_frame_interval)  # Returns a uniformly-timed (W,H,N) array of frames

    '''Create the DSA object'''
    dsa = DSASequence(frames, frame_interval=args.desired_frame_interval)

    '''Sample the Arterial Input Function'''
    (_,aif_mask) = dsa.sample_AIF_from_perfDSA(args.model_path, device=args.device)

    '''Plot the and save the results'''
    minip = cv.cvtColor(dsa.MINIP(), cv.COLOR_GRAY2BGR).astype(np.uint8)
    contours, _ = cv.findContours(aif_mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    AIF_contour_points = max(contours, key=cv.contourArea)
    cv.drawContours(minip, AIF_contour_points, -1, (255, 0, 0), thickness=2)
    
    fig, ax = plt.subplots(2,3, figsize=(16,10), tight_layout=True)
    im = np.empty_like(ax, dtype=object)
    ax[0,0].plot(dsa.Time(),dsa.AIF())
    ax[0,0].set(title="AIF", ylabel="Dye concentration [a.u.]", xlabel="Time (s)")
    
    im[1,0] = ax[1,0].imshow(minip)
    ax[1,0].set(title="MINIP",xticks=[],yticks=[])
    
    im[0,1] = ax[0,1].imshow(dsa.CBV(),cmap='jet')
    ax[0,1].set(title="CBV",xticks=[],yticks=[])
    fig.colorbar(im[0,1], ax=ax[0,1], orientation='vertical', fraction=0.046, pad=0.04)
    
    im[1,1] = ax[1,1].imshow(dsa.CBF(),cmap='jet')
    ax[1,1].set(title="CBF",xticks=[],yticks=[])
    fig.colorbar(im[1,1], ax=ax[1,1], orientation='vertical', fraction=0.046, pad=0.04)
    
    im[0,2] = ax[0,2].imshow(dsa.MTT(),cmap='jet')
    ax[0,2].set(title="MTT",xticks=[],yticks=[])
    fig.colorbar(im[0,2], ax=ax[0,2], orientation='vertical', fraction=0.046, pad=0.04)
    
    im[1,2] = ax[1,2].imshow(dsa.Tmax(),cmap='jet')
    ax[1,2].set(title="Tmax",xticks=[],yticks=[])
    fig.colorbar(im[1,2], ax=ax[1,2], orientation='vertical', fraction=0.046, pad=0.04)
    
    fig.suptitle('Perfusion parameters')
    plt.savefig(args.out_img, dpi=300)

    logging.info('Perfusion maps written to {:}'.format(args.out_img))
    logging.info("Done!")