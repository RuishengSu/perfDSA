import logging
import os
from pathlib import Path

import pandas as pd

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class StudyRepository:
    study_csv_path = '/mnt/data2/info/studies.csv'
    study_pk = ['study_id', 'patient_id']
    col_names = ['study_id', 'patient_id', 'study_date', 'study_time', 'annotation', 'path']

    Path(study_csv_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(study_csv_path):
        df_study = pd.read_csv(study_csv_path)
    else:
        df_study = pd.DataFrame(columns=col_names)
    '''add column into CSV if newly defined in col_names'''
    for column in col_names:
        if column not in df_study:
            df_study[column] = ""
    df_study = df_study.set_index(study_pk)

    @classmethod
    def upsert_row(cls, row_dict, to_csv=False):
        df_row = pd.DataFrame([row_dict], columns=row_dict.keys()).set_index(cls.study_pk)
        cls.df_study = pd.concat([cls.df_study, df_row[~df_row.index.isin(cls.df_study.index)]])
        cls.df_study.update(df_row)
        if to_csv:
            cls.to_csv()

    @classmethod
    def update_row(cls, row, index=None, to_csv=False):
        if not isinstance(row, pd.DataFrame):
            df_row = pd.DataFrame([row], columns=row.keys())
        else:
            df_row = row
        if not index:
            index = cls.study_pk
        df_row = df_row.reset_index().set_index(index)

        cls.df_study = cls.df_study.reset_index().set_index(index)
        if df_row[df_row.index.isin(cls.df_study.index)].empty:
            logger.error("Did not find a matching entry to update.")
            raise ValueError
        else:
            cls.df_study.update(df_row)
        if to_csv:
            cls.to_csv()

        cls.df_study = cls.df_study.reset_index().set_index(cls.study_pk)

    @classmethod
    def get_dataframe(cls):
        return cls.df_study.reset_index()

    @classmethod
    def to_csv(cls):
        # hdr = False if os.path.isfile(cls.study_csv_path) else True
        # cls.df_study.to_csv(cls.study_csv_path, mode='a', header=hdr)
        # cls.df_study.set_index('study_id')
        cls.df_study.to_csv(cls.study_csv_path)

    @classmethod
    def delete_patient(cls, patient_id):
        cls.df_study = cls.df_study[cls.df_study.patient_id != patient_id]
        cls.to_csv()

    @classmethod
    def delete_row(cls, row_dict):
        df_to_be_deleted = cls.df_study
        for key, value in row_dict.items():
            df_to_be_deleted = df_to_be_deleted[df_to_be_deleted[key] == value]
        cls.df_study = cls.df_study.drop(df_to_be_deleted.index)
        cls.to_csv()
