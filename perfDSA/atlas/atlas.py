import glob
import os
from pathlib import Path

import cv2 as cv
import nibabel as nib
import numpy as np

import phase_classification.utils
from baseline import pipeline
from utils import utils

input_dir = "atlas/original"
output_sequence_dir = "atlas/sequences/RMVeinYes"
output_minip_dir = "atlas/minips/RMVeinYes"
Path(output_minip_dir).mkdir(parents=True, exist_ok=True)

orignal_sequence_paths = [f for f in glob.glob(input_dir + "/*", recursive=False) if os.path.isfile(f)]

for sequence_path in orignal_sequence_paths:
    print(sequence_path)
    img_sequence, _, _ = utils.load_itkfile(sequence_path)
    img_sequence, _ = pipeline.remove_venous_phase(img_sequence)

    img_sequence_nii = np.transpose(img_sequence, (2, 1, 0))
    img_sequence_nii = nib.Nifti1Image(img_sequence_nii, np.eye(4))
    output_sequence_path = sequence_path.replace(input_dir, output_sequence_dir)
    Path(output_sequence_dir).parent.mkdir(parents=True, exist_ok=True)
    nib.save(img_sequence_nii, output_sequence_path)

    img_sequence = pipeline.elastix_align_sequence(img_sequence)
    img_minip = utils.minip(img_sequence)
    img_minip = phase_classification.utils.normalize(img_minip)
    output_minip_path = sequence_path.replace(input_dir, output_minip_dir) + '.bmp'
    cv.imwrite(output_minip_path, img_minip)
