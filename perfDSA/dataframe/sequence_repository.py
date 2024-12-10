import logging
import os
from pathlib import Path

import pandas as pd

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceRepository:

    def __init__(self, sequence_csv_path='/mnt/data2/info/sequences.csv'):
        self.sequence_pk = ['sequence_name']
        self.col_names = ['patient_id', 'sequence_name', 'view', 'pre_post_EVT', 'series_number', 'number_of_frames',
                          'arterial_phase_first_frame_annotated_by_ruisheng',
                          'arterial_phase_last_frame_annotated_by_ruisheng',
                          'parenchymal_phase_last_frame_annotated_by_ruisheng',
                          'venous_phase_last_frame_annotated_by_ruisheng',
                          'arterial_phase_first_frame_annotated_by_theo',
                          'arterial_phase_last_frame_annotated_by_theo',
                          'parenchymal_phase_last_frame_annotated_by_theo',
                          'venous_phase_last_frame_annotated_by_theo',
                          'arterial_phase_first_frame_annotated_by_sandra',
                          'arterial_phase_last_frame_annotated_by_sandra',
                          'parenchymal_phase_last_frame_annotated_by_sandra',
                          'venous_phase_last_frame_annotated_by_sandra',
                          'arterial_phase_first_frame_consensus',
                          'arterial_phase_last_frame_consensus',
                          'parenchymal_phase_last_frame_consensus',
                          'venous_phase_last_frame_consensus',
                          'selected_for_autoTICI_scoring',
                          'selected_for_phase_classification_testing',
                          'selected_for_phase_classification_training_subset_ruisheng',
                          'selected_for_phase_classification_training_subset_sandra',
                          'selected_for_phase_classification_training_subset_theo']

        self.sequence_csv_path = sequence_csv_path
        if os.path.isfile(self.sequence_csv_path):
            self.df_sequence = pd.read_csv(sequence_csv_path, dtype={'series_number': int,
                                                                     'number_of_frames': int}).set_index(
                self.sequence_pk)
            self.df_sequence = self.df_sequence.reset_index()
            for column in self.col_names:
                if column not in self.df_sequence:
                    self.df_sequence[column] = ""
            self.df_sequence = self.df_sequence.set_index(self.sequence_pk)
        else:
            Path(self.sequence_csv_path).parent.mkdir(parents=True, exist_ok=True)
            self.df_sequence = pd.DataFrame(columns=self.col_names).set_index(
                self.sequence_pk)

    def upsert_row(self, row_dict, to_csv=False):
        df_row = pd.DataFrame([row_dict], columns=row_dict.keys()).set_index(self.sequence_pk)
        self.df_sequence = pd.concat([self.df_sequence, df_row[~df_row.index.isin(self.df_sequence.index)]])
        self.df_sequence.update(df_row)
        if to_csv:
            self.to_csv()

    def update_row(self, row, index=None, to_csv=True):
        if not isinstance(row, pd.DataFrame):
            df_row = pd.DataFrame([row], columns=row.keys())
        else:
            df_row = row

        if not index:
            index = self.sequence_pk
        df_row = df_row.set_index(index)

        self.df_sequence = self.df_sequence.reset_index().set_index(index)

        if df_row[df_row.index.isin(self.df_sequence.index)].empty:
            logger.error("Did not find a matching entry to update.")
            raise ValueError
        else:
            self.df_sequence.update(df_row)
        if to_csv:
            self.to_csv()

        self.df_sequence = self.df_sequence.reset_index().set_index(self.sequence_pk)

    def get_dataframe(self):
        return self.df_sequence.reset_index()

    def to_csv(self):
        self.df_sequence.to_csv(self.sequence_csv_path)

    def delete_patient(self, patient_id):
        self.df_sequence = self.df_sequence[self.df_sequence.patient_id != patient_id]
        self.to_csv()

    def delete_row(self, row_dict):
        df_to_be_deleted = self.df_sequence
        for key, value in row_dict.items():
            df_to_be_deleted = df_to_be_deleted[df_to_be_deleted[key] == value]
        self.df_sequence = self.df_sequence.drop(df_to_be_deleted.index)
        self.to_csv()
