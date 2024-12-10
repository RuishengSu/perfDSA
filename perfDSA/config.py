import os
from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")


"""Parameters"""
remove_non_contrast_frame_enabled = False
remove_text_and_black_border_enabled = False
motion_correction_enabled = False
pre_post_sequence_registration_enabled = True
denoising_enabled = False
time_regularization_enabled = True

reuse_text_free_images = False
reuse_motion_correction_results = False
reuse_pre_post_registration_results = True
reuse_denoised_sequence = False
reuse_time_regularization_results = True
reuse_irf_sequence_results = True


noise_level = 10  # pixel density of 10 within 0-255 range
MAX_VALID_INTENSITY = 200  # uint8
MIN_VESSEL_INTENSITY = 2
MIN_PERFUSION_INTENSITY = 10
FRONGI_INTENSITY_THRES = 20
frame_interval = 250  # desired frame time interval in ms for resampling of time intensity curves
wiener_deconv_lambda = 5

"""Input"""
orig_angio_dir = '/media/ruisheng/Data4/swine/exp_20211207/original/2D'
subt_angio_dir = '/mnt/data2/Dropbox/Ruisheng/PhD/Collaborations/swine angiography/DSA files Opus 64-66'
perf_angio_dir = '/media/ruisheng/Data4/swine/exp_20220220/vis_perfusion'

ica_mask_dir = './../mevislab/ICA_annotation/results'
mask_file_name_suffix = '-mask-ica.bmp'

"""Output paths"""
# output_path_base = '/mnt/data1/timeinfo/output'
# current_result_dir = os.path.join(output_path_base, 'result_{}'.format(now))
# latest_result_dir = os.path.join(output_path_base, 'result_latest')
#
# log_filepath = os.path.join(current_result_dir, 'timeinfo.log')
# current_paramap_dirpath = os.path.join(current_result_dir, 'vis_paraMaps')
# current_vis_sequence_registration_dirpath = os.path.join(current_result_dir, 'vis_sequence_registration')
# current_vis_superpixel_dirpath = os.path.join(current_result_dir, 'vis_superpixel')
# current_vis_tic_dirpath = os.path.join(current_result_dir, 'vis_tic')
# current_vis_aif_dirpath = os.path.join(current_result_dir, 'vis_aif')
# current_vis_deconvolution_dirpath = os.path.join(latest_result_dir, 'vis_deconvolution')
#
# latest_log_filepath = os.path.join(latest_result_dir, 'timeinfo.log')
# latest_paramap_dirpath = os.path.join(latest_result_dir, 'vis_paraMaps')
# latest_vis_sequence_registration_dirpath = os.path.join(latest_result_dir, 'vis_sequence_registration')
# latest_vis_superpixel_dirpath = os.path.join(latest_result_dir, 'vis_superpixel')
# latest_vis_tic_dirpath = os.path.join(latest_result_dir, 'vis_tic')
# latest_vis_aif_dirpath = os.path.join(latest_result_dir, 'vis_aif')
# latest_vis_deconvolution_dirpath = os.path.join(latest_result_dir, 'vis_deconvolution')
#
# latest_result_reuse_dir = os.path.join(latest_result_dir, 'reuse')
# latest_text_free_sequence_path = os.path.join(latest_result_reuse_dir, 'textfree_sequences')
# latest_mc_sequence_dirpath = os.path.join(latest_result_reuse_dir, 'mc_sequences')
# latest_registered_preEVT_dirpath = os.path.join(latest_result_reuse_dir, 'registered_sequences')
# latest_denoised_sequence_dirpath = os.path.join(latest_result_reuse_dir, 'denoised_sequences')
# latest_time_regularization_dirpath = os.path.join(latest_result_reuse_dir, 'time_regularized_sequences')
# latest_irf_sequence_dirpath = os.path.join(latest_result_reuse_dir, 'offset_corrected_sequences')
