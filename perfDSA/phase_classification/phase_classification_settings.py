from pathlib import Path

num_classes = 4
# number_of_epochs = 100
model_input_size = 224
# batch_size = 32
# random_seed = 42
# learning_rate = 0.01
# cv_fold = 5

"""Configure input paths"""
# training_dataset_path = '/mnt/datassd/dsa_pc/training'
# test_dataset_path = '/mnt/datassd/dsa_pc/testing'

"""Configure output paths"""
# output_dirpath = "phase_classification/outputs"
# intermediate_dirpath = output_dirpath + '/intermediate'
# frame_offset_output_pickle_path = intermediate_dirpath + '/phase_pred_ann_diff.pickle'

"""Model paths"""
best_phase_model_path = "phase_classification/multi_frame_best.model"
# self_trained_resnet_model_for_lstm = model_output_dirpath + "/self_trained_model_as_lstm_input.model"

"""Configure output visualization paths"""
# output_fig_dirpath = output_dirpath + '/fig'
# learning_curve_fig_output_path = output_fig_dirpath + '/learning_curve_multi_frame.svg'
# normalized_confusion_matrix_fig_output_path = output_fig_dirpath + '/cm_normalized_multi_frame.svg'
# confusion_matrix_fig_output_path = output_fig_dirpath + '/cm_multi_frame.svg'

"""Log file path"""
# log_filepath = "log/interobserver2.log"
# text_free_sequence_dirpath = '/mnt/data1/autoTICI/output/phase_classification/text_free_sequences'
# Path(text_free_sequence_dirpath).mkdir(parents=True, exist_ok=True)
