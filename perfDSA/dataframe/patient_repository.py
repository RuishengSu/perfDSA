import logging
import os
from pathlib import Path

import pandas as pd

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientRepository:
    patient_csv_path = '/mnt/data2/info/patients.csv'
    Path(patient_csv_path).parent.mkdir(parents=True, exist_ok=True)
    patient_pk = 'patient_id'
    col_names = ['patient_id', 'hemisphere', 'occ_loc',
                 'pre_tici', 'post_tici', 'imputed_post_tici', 'mrs', 'imputed_mrs_rev', 'NIHSS_BL', 'NIHSS_FU',
                 'NIHSS_diff', 'annotation', 'path']
    if os.path.isfile(patient_csv_path):
        df_patient = pd.read_csv(patient_csv_path).set_index(patient_pk)

        df_patient = df_patient.reset_index()
        for column in col_names:
            if column not in df_patient:
                df_patient[column] = ""
        df_patient = df_patient.set_index(patient_pk)

    else:
        df_patient = pd.DataFrame(columns=col_names).set_index(patient_pk)

    @classmethod
    def upsert_row(cls, row, index=None, to_csv=False, add_column=False):
        if not isinstance(row, pd.DataFrame):
            df_row = pd.DataFrame([row], columns=row.keys())
        else:
            df_row = row

        if not index:
            index = cls.patient_pk
        df_row = df_row.set_index(index)

        cls.df_patient = cls.df_patient.reset_index().set_index(index)
        if add_column:
            for column in df_row.keys():
                if column not in cls.df_patient:
                    cls.df_patient[column] = ""

        cls.df_patient = pd.concat([cls.df_patient, df_row[~df_row.index.isin(cls.df_patient.index)]])
        cls.df_patient.update(df_row)
        if to_csv:
            cls.to_csv()

        cls.df_patient = cls.df_patient.reset_index().set_index(cls.patient_pk)

    @classmethod
    def update_row(cls, row, index=None, to_csv=False):
        if not isinstance(row, pd.DataFrame):
            df_row = pd.DataFrame([row], columns=row.keys())
        else:
            df_row = row
        if not index:
            index = cls.patient_pk
        df_row = df_row.set_index(index)

        cls.df_patient = cls.df_patient.reset_index().set_index(index)

        if df_row[df_row.index.isin(cls.df_patient.index)].empty:
            logger.error("Did not find a matching entry to update.")
            raise ValueError()
        else:
            cls.df_patient.update(df_row)
        if to_csv:
            cls.to_csv()
        cls.df_patient = cls.df_patient.reset_index().set_index(cls.patient_pk)

    @classmethod
    def get_dataframe(cls):
        return cls.df_patient.reset_index()

    @classmethod
    def to_csv(cls):
        # hdr = False if os.path.isfile(cls.study_csv_path) else True
        # cls.df_study.to_csv(cls.study_csv_path, mode='a', header=hdr)
        # cls.df_patient.set_index('patient_id')
        cls.df_patient.to_csv(cls.patient_csv_path)

    @classmethod
    def delete_patient(cls, patient_id):
        cls.df_patient = cls.df_patient[cls.df_patient.patient_id != patient_id]
        cls.to_csv()

    @classmethod
    def get_hemisphere(cls, patient_id):
        hemishpere = cls.df_patient.loc[patient_id, 'hemisphere']
        if 'Right' in hemishpere:
            return 'right'
        if 'Left' in hemishpere:
            return 'left'
        raise ValueError

    @classmethod
    def get_eTICI(cls, patient_id):
        return cls.df_patient.loc[patient_id, 'post_tici']

    @classmethod
    def get_mRS(cls, patient_id):
        return cls.df_patient.loc[patient_id, 'mrs']

    @classmethod
    def get_imputed_mRS_rev(cls, patient_id):
        return cls.df_patient.loc[patient_id, 'imputed_mrs_rev']

    @classmethod
    def get_NIHSS_diff(cls, patient_id):
        return cls.df_patient.loc[patient_id, 'NIHSS_diff']

    @classmethod
    def get_NIHSS_BL(cls, patient_id):
        return cls.df_patient.loc[patient_id, 'NIHSS_BL']

    @classmethod
    def get_NIHSS_FU(cls, patient_id):
        return cls.df_patient.loc[patient_id, 'NIHSS_FU']

    @classmethod
    def get_occ_loc(cls, patient_id):
        return cls.df_patient.loc[patient_id, 'occ_loc']
