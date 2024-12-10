import os
from datetime import datetime
from pathlib import Path

now = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")

"""Setting parameters"""
remove_text_and_black_border = True
motion_correction_enabled = True
landmark_preregistration_enabled = True
remove_venous_phase = True
respacing_enabled = True

reuse_motion_correction_results = True
reuse_landmark = True
reuse_registration = True
reuse_per_patient_results = False
reuse_overall_results = False

"""CONSTANTS"""
VESSEL = 2
PERFUSED = 1
NON_PERFUSED = 0
MAX_VALID_INTENSITY = 200  # uint8
MIN_VESSEL_INTENSITY = 2
MIN_PERFUSION_INTENSITY = 10
FRONGI_INTENSITY_THRES = 20

"""Input"""
patient_info_filepath = '/mnt/data2/3.sequences/mrclean.csv'
mrclean_dicom_path = '/mnt/data2/3.sequences/mrclean'
clean_dicom_path = '/mnt/data2/3.sequences/mrclean'
# clean_dicom_path = '/media/ruisheng/Data4/outcome_prediction/mrclean'
# clean_dicom_path = '/mnt/data1/TIC_NoIV_yvonne/selected_dsa/'
landmark_detection_dir = './landmark_detection'
atlas_dir = './atlas'
df_atlas_path = os.path.join(atlas_dir, 'atlas.csv')

"""Output"""
output_dir = '/home/ftenijenhuis/Documents/perf_test_output'

log_filepath = os.path.join(output_dir, 'pipeline_{}.log'.format(now))
output_vis_dir = os.path.join(output_dir, 'vis')
output_tics_dir = os.path.join(output_dir, 'tics')
output_ica_vessel_dir = os.path.join(output_dir, 'ica_vessel')
output_vessel_dir = os.path.join(output_dir, 'vessel')
output_csv_path = os.path.join(output_dir, 'results.csv')
output_reusable_dirpath = os.path.join(output_dir, 'reuse')
mc_sequence_elastix_path = os.path.join(output_reusable_dirpath, 'mc_sequences_elastix')
landmark_dirpath = os.path.join(output_reusable_dirpath, 'landmark')
registration_dirpath = os.path.join(output_reusable_dirpath, 'registration')
per_patient_result_dirpath = os.path.join(output_reusable_dirpath, 'tic_results')

"""Initialization"""
Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(output_vis_dir).mkdir(exist_ok=True, parents=True)
Path(output_tics_dir).mkdir(exist_ok=True, parents=True)
Path(mc_sequence_elastix_path).mkdir(parents=True, exist_ok=True)
Path(landmark_dirpath).mkdir(parents=True, exist_ok=True)
Path(registration_dirpath).mkdir(parents=True, exist_ok=True)
Path(output_reusable_dirpath).mkdir(exist_ok=True, parents=True)
Path(per_patient_result_dirpath).mkdir(exist_ok=True, parents=True)
