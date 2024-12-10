import logging

import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import transforms
from phase_classification import models
from phase_classification import phase_classification_settings
from phase_classification.utils import label_fit

logger = logging.getLogger(__name__)

# Define transformations for the validation and test set
transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(phase_classification_settings.model_input_size),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, ), (0.5, )),
])

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()
device = models.get_device()
model = models.init_model()
model = models.load_model_state_dict(model, model_state_dict_path=phase_classification_settings.best_phase_model_path)


def predict_sequence_phases(sequence_array):
    """:sequence_array: a 2D+t sequence in 3D array format. Time axis is expected to be first channel."""
    model.eval()
    sequence_as_model_input = []
    sequence_length = sequence_array.shape[0]
    for idx_frame in range(sequence_length):
        current_frame = sequence_array[idx_frame]
        previous_frame = sequence_array[max(0, idx_frame - 1)]
        next_frame = sequence_array[min(sequence_length - 1, idx_frame + 1)]

        image_input = np.dstack([previous_frame, current_frame, next_frame])

        '''Normalize image if input is not uint8'''
        if not isinstance(image_input[0, 0, 0], np.uint8):
            image_input = cv.normalize(image_input, None, 0, 255, cv.NORM_MINMAX)
            image_input = image_input.astype(np.uint8)

        image_input = transformations(image_input)
        sequence_as_model_input.append(image_input)
    sequence_as_model_input = torch.stack(sequence_as_model_input)

    if cuda_avail:
        sequence_as_model_input = sequence_as_model_input.to(device)
    with torch.no_grad():
        outputs = model(sequence_as_model_input)
    predicted_sequence_phases = label_fit(outputs)
    logger.info("Predicted sequence phases: {}".format(predicted_sequence_phases))

    return predicted_sequence_phases


if __name__ == "__main__":
    print("Done!")
