import logging
import os
from pathlib import Path

import pandas as pd

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class PhaseRepository:
    phase_csv_path = '/mnt/data2/info/phase.csv'
    series_pk = ['patient_id', 'filename']
    Path(phase_csv_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(phase_csv_path):
        df_phase = pd.read_csv(phase_csv_path, dtype={'series_number': int, 'number_of_frames': int}).set_index(
            series_pk)
    else:
        col_names = ['patient_id', 'filename', 'series_number', 'view', 'number_of_frames',
                     'arterial_phase_first_frame', 'arterial_phase_last_frame', 'parenchymal_phase_last_frame',
                     'venous_phase_last_frame']
        df_phase = pd.DataFrame(columns=col_names).set_index(series_pk)

    @classmethod
    def upsert_row(cls, row_dict, to_csv=False):
        df_row = pd.DataFrame([row_dict], columns=row_dict.keys()).set_index(cls.series_pk)
        cls.df_phase = pd.concat([cls.df_phase, df_row[~df_row.index.isin(cls.df_phase.index)]])
        cls.df_phase.update(df_row)
        if to_csv:
            cls.to_csv()

    @classmethod
    def update_row(cls, row, index=None, to_csv=True):
        if not isinstance(row, pd.DataFrame):
            df_row = pd.DataFrame([row], columns=row.keys())
        else:
            df_row = row
        if not index:
            index = cls.series_pk
        df_row = df_row.set_index(index)
        cls.df_phase = cls.df_phase.reset_index().set_index(index)
        if df_row[df_row.index.isin(cls.df_phase.index)].empty:
            logger.error("Did not find a matching entry to update.")
            raise ValueError
        else:
            cls.df_phase.update(df_row)
        if to_csv:
            cls.to_csv()

        cls.df_phase = cls.df_phase.reset_index().set_index(cls.series_pk)

    @classmethod
    def get_dataframe(cls):
        return cls.df_phase.reset_index()

    @classmethod
    def to_csv(cls):
        cls.df_phase.to_csv(cls.phase_csv_path)

    @classmethod
    def delete_patient(cls, patient_id):
        cls.df_phase = cls.df_phase[cls.df_phase.patient_id != patient_id]
        cls.to_csv()

    @classmethod
    def delete_row(cls, row_dict):
        df_to_be_deleted = cls.df_phase
        for key, value in row_dict.items():
            df_to_be_deleted = df_to_be_deleted[df_to_be_deleted[key] == value]
        cls.df_phase = cls.df_phase.drop(df_to_be_deleted.index)
        cls.to_csv()
