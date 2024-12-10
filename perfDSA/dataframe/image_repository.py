import logging
import os
from pathlib import Path

import pandas as pd

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageRepository:
    image_csv_path = '/mnt/data2/info/images.csv'
    Path(image_csv_path).parent.mkdir(parents=True, exist_ok=True)
    image_pk = ['image_id', 'patient_id']
    if os.path.isfile(image_csv_path):
        df_image = pd.read_csv(image_csv_path).set_index(image_pk)
    else:
        col_names = ['image_id', 'instance_number', 'is_multiframe', 'number_of_frames', 'image_date', 'image_time',
                     'rows', 'columns', 'series_id', 'series_view', 'study_id', 'patient_id', 'annotation', 'path',
                     'bad_frames']
        df_image = pd.DataFrame(columns=col_names).set_index(image_pk)

    @classmethod
    def upsert_row(cls, row_dict, to_csv=False):
        df_row = pd.DataFrame(row_dict).set_index(cls.image_pk)
        cls.df_image = pd.concat([cls.df_image, df_row[~df_row.index.isin(cls.df_image.index)]])
        cls.df_image.update(df_row)
        if to_csv:
            cls.to_csv()

    @classmethod
    def update_row(cls, row, index=None, to_csv=False, add_column=False):
        if not isinstance(row, pd.DataFrame):
            df_row = pd.DataFrame([row], columns=row.keys())
        else:
            df_row = row

        if not index:
            index = cls.image_pk
        df_row = df_row.set_index(index)

        cls.df_image = cls.df_image.reset_index().set_index(index)
        if add_column:
            for column in df_row.keys():
                if column not in cls.df_image:
                    cls.df_image[column] = ""
        if df_row[df_row.index.isin(cls.df_image.index)].empty:
            logger.error("Did not find a matching entry to update.")
            raise ValueError
        else:
            cls.df_image.update(df_row)
        if to_csv:
            cls.to_csv()
        cls.df_image = cls.df_image.reset_index().set_index(cls.image_pk)

    @classmethod
    def get_dataframe(cls):
        return cls.df_image.reset_index()

    @classmethod
    def to_csv(cls):
        # hdr = False if os.path.isfile(cls.study_csv_path) else True
        # cls.df_study.to_csv(cls.study_csv_path, mode='a', header=hdr)
        # cls.df_image.set_index('image_id')
        cls.df_image.to_csv(cls.image_csv_path)

    @classmethod
    def delete_patient(cls, patient_id):
        cls.df_image = cls.df_image[cls.df_image.patient_id != patient_id]
        cls.to_csv()

    @classmethod
    def delete_row(cls, row_dict):
        df_to_be_deleted = cls.df_image
        for key, value in row_dict.items():
            df_to_be_deleted = df_to_be_deleted[df_to_be_deleted[key] == value]
        cls.df_image = cls.df_image.drop(df_to_be_deleted.index)
        cls.to_csv()
