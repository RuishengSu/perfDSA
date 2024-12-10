import logging
import os
from pathlib import Path

import pandas as pd

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class SeriesRepository:
    series_csv_path = '/mnt/data2/info/series.csv'
    series_pk = ['series_id', 'patient_id']
    Path(series_csv_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(series_csv_path):
        df_series = pd.read_csv(series_csv_path, dtype={'series_number': int, 'number_of_frames': int}).set_index(
            series_pk)
    else:
        col_names = ['series_id', 'series_number', 'view', 'description', 'series_date', 'series_time', 'system',
                     'resolution', 'number_of_frames', 'study_id', 'patient_id', 'annotation', 'path']
        df_series = pd.DataFrame(columns=col_names).set_index(series_pk)

    @classmethod
    def upsert_row(cls, row_dict, to_csv=False):
        df_row = pd.DataFrame(row_dict).set_index(cls.series_pk)
        cls.df_series = pd.concat([cls.df_series, df_row[~df_row.index.isin(cls.df_series.index)]])
        cls.df_series.update(df_row)
        if to_csv:
            cls.to_csv()

    @classmethod
    def update_row(cls, row, index=None, to_csv=True, add_column=False):
        if not isinstance(row, pd.DataFrame):
            df_row = pd.DataFrame([row], columns=row.keys())
        else:
            df_row = row

        if not index:
            index = cls.series_pk
        df_row = df_row.set_index(index)

        cls.df_series = cls.df_series.reset_index().set_index(index)
        if add_column:
            for column in df_row.keys():
                if column not in cls.df_series:
                    cls.df_series[column] = ""

        if df_row[df_row.index.isin(cls.df_series.index)].empty:
            logger.error("Did not find a matching entry to update.")
            raise ValueError
        else:
            cls.df_series.update(df_row)
        if to_csv:
            cls.to_csv()

        cls.df_series = cls.df_series.reset_index().set_index(cls.series_pk)

    @classmethod
    def get_dataframe(cls):
        return cls.df_series.reset_index()

    @classmethod
    def to_csv(cls):
        # hdr = False if os.path.isfile(cls.study_csv_path) else True
        # cls.df_study.to_csv(cls.study_csv_path, mode='a', header=hdr)
        # cls.df_series.set_index(['series_id', 'view'])
        cls.df_series.to_csv(cls.series_csv_path)

    @classmethod
    def delete_patient(cls, patient_id):
        cls.df_series = cls.df_series[cls.df_series.patient_id != patient_id]
        cls.to_csv()

    @classmethod
    def delete_row(cls, row_dict):
        df_to_be_deleted = cls.df_series
        for key, value in row_dict.items():
            df_to_be_deleted = df_to_be_deleted[df_to_be_deleted[key] == value]
        cls.df_series = cls.df_series.drop(df_to_be_deleted.index)
        cls.to_csv()
