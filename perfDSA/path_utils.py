import numpy as np
import logging
from skimage.morphology import binary_dilation, disk, remove_small_objects
import config
import cv2 as cv
logger = logging.getLogger(__name__)


def get_mrclean_subject_id(filename):
    import re
    try:
        return re.search('(R[0-9]{4}?)', filename).group(1)
    except AttributeError as error:
        logger.error('Error while getting subject id from filepath: {}'.format(filename))
        logger.error('\n' + str(error))
        return None


def get_noiv_subject_id(filename):
    import re
    try:
        return re.search('(mrcleannoiv_[0-9]{5}_I1_DSA?)', filename).group(1)
    except AttributeError as error:
        logger.error('Error while getting subject id from filepath: {}'.format(filename))
        logger.error('\n' + str(error))
        return None


def minip(img_seq, axis=0):
    return np.min(img_seq, axis=axis)


def extract_patient_id_from_path(filename):
    import re
    try:
        return re.search('(R[0-9]{4}?)', filename).group(1)
    except AttributeError as error:
        logger.error('Error while getting subject id from filepath: {}'.format(filename))
        logger.error('\n' + str(error))
        return None


def remove_text_and_border(numpy_array):
    def remove_text_2d(img, text_inpaint_radius=3, border_inpaint_radius=10, border_margin=3):
        """
        Input: expected image range 0-255; if 2D+t, t is expected to be the first channel.
        Remove text and black border lines from a 2D image
        border_margin: border line dilation radius"""
        '''text mask'''
        black_text_mask = np.zeros([*img.shape], dtype=bool)
        black_text_mask[img < config.MIN_VESSEL_INTENSITY] = True
        black_text_mask = binary_dilation(black_text_mask, disk(10))

        white_text_mask = np.zeros([*img.shape], dtype=bool)
        white_text_mask[img > config.MAX_VALID_INTENSITY] = True
        white_text_mask = binary_dilation(white_text_mask, disk(10))

        text_mask = white_text_mask & black_text_mask
        text_mask = binary_dilation(text_mask, disk(5))

        '''black border line mask'''
        black_border_mask = np.zeros(img.shape, dtype=bool)
        black_border_mask[img < config.MIN_VESSEL_INTENSITY] = True
        black_border_mask = remove_small_objects(black_border_mask, min_size=500, connectivity=2)
        black_border_mask = binary_dilation(black_border_mask, disk(border_margin))

        '''Combined mask'''
        mask = (text_mask | black_border_mask)
        '''inpaint with small radius for less blur in text area'''
        img = cv.inpaint(img, mask.astype(np.uint8), text_inpaint_radius, flags=cv.INPAINT_TELEA)
        '''inpaint with larger radius for less noise in border area'''
        img = cv.inpaint(img, black_border_mask.astype(np.uint8), border_inpaint_radius, flags=cv.INPAINT_TELEA)
        return img

    dim = len(numpy_array.shape)
    if dim == 3:
        for idx, frame in enumerate(numpy_array):
            numpy_array[idx] = remove_text_2d(frame)
    elif dim == 2:
        numpy_array = remove_text_2d(numpy_array)
    return numpy_array
