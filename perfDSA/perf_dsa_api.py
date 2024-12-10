import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import cv2 as cv
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
from tensorflow.keras.models import load_model

import perfDSA.neurite_plot as ne
import perfDSA.pipeline_settings as settings
from perfDSA import configs, utils
from perfDSA.non_parametric_deconvolution import modelfree_deconv
from perfDSA.main_TIC import normalize, get_cum_time_vector, temporal_interp, plot_tics, detect_landmarks, analyse_TICs

logger = logging.getLogger(__name__)

def main(args):
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
                series_idx + 1, df_selection.shape[0], utils.get_mrclean_subject_id(series_path),
                Path(series_path).stem))
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
    # df_sequence = pd.read_csv("../csv/230712-tic_selection_with_venous_2.csv")
    # Implemented a fix from https://github.com/keras-team/keras/issues/19441 in the site-packages
    model_ap = load_model(os.path.join(settings.landmark_detection_dir, 'ap_combined.h5'), compile=False)
    model_lateral = load_model(os.path.join(settings.landmark_detection_dir, 'lateral_combined.h5'), compile=False)


