import logging
from itertools import combinations_with_replacement

import torch
import torch.nn as nn

from phase_classification import phase_classification_settings, models

logger = logging.getLogger(__name__)
device = models.get_device()
validation_loss_fn = nn.CrossEntropyLoss(reduction='sum')


def postprocess_sequence_phases(sequence_phases):
    for idx_frame in range(1, len(sequence_phases) - 1):
        phase_of_current_frame = sequence_phases[idx_frame]
        phase_of_previous_frame = sequence_phases[idx_frame - 1]
        phase_of_next_frame = sequence_phases[idx_frame + 1]
        if phase_of_current_frame not in [phase_of_previous_frame, phase_of_next_frame]:
            if phase_of_previous_frame == phase_of_next_frame:
                sequence_phases[idx_frame] = phase_of_previous_frame
    return sequence_phases


def extract_phase_separation_indices(sequence_phases):
    try:
        arterial_phase_first_frame = sequence_phases.index(1)
        arterial_phase_last_frame = len(sequence_phases) - sequence_phases[::-1].index(1) - 1
    except ValueError:
        arterial_phase_first_frame, arterial_phase_last_frame = None, None

    try:
        parenchymal_phase_last_frame = len(sequence_phases) - sequence_phases[::-1].index(2) - 1
    except ValueError:
        parenchymal_phase_last_frame = None
    return arterial_phase_first_frame, arterial_phase_last_frame, parenchymal_phase_last_frame


def label_fit(label_softmax_prob_sequence):
    lbl_length = label_softmax_prob_sequence.shape[0]
    """Generate all possible valid label sequences"""
    combs = []
    for comb in combinations_with_replacement(range(phase_classification_settings.num_classes + 1), lbl_length):
        valid = True
        comb = list(comb)
        for i in range(1, len(comb)):
            if not (comb[i - 1] <= comb[i] <= comb[i - 1] + 1):
                valid = False
                break
        if valid:
            comb = [0 if l == 4 else l for l in comb]
            combs.append(comb)

    """If input label list is a valid, return"""
    _, label_sequence = torch.max(label_softmax_prob_sequence, 1)
    label_sequence = label_sequence.tolist()
    if label_sequence in combs:
        return label_sequence

    """"Otherwise, find the closest valid label sequence"""
    min_loss = None
    output = label_sequence
    for comb in combs:
        loss = validation_loss_fn(label_softmax_prob_sequence, torch.tensor(comb).to(device)).item()
        if (min_loss is None) or (min_loss > loss):
            min_loss = loss
            output = comb
    logger.info("\nwithout sequence labelling logic: {}\n"
                "with sequence labelling logic:    {}".format(label_sequence, output))
    return output
