from enum import Enum


class DSAClassBase:

    def __init__(self, uid, path=None):
        self.id = uid
        self.path = path

    def set_path(self, path):
        self.path = path


class ExtendedEnum(Enum):
    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class Image_Annotation_Enum(ExtendedEnum):
    OK = 'ok'
    CORRUPTED = 'corrupted'
    DUPLICATED = 'duplicated'
    NON_XA = 'not XA'
    UNSPECIFIED = 'unspecified'
    UNSUBTRACTED = 'unsubtracted'
    BAD = 'bad quality'


class Study_Annotation_Enum(ExtendedEnum):
    OK = 'ok'
    # DUPLICATED = 'duplicated with a study under another patient'
    OBSOLETE = 'obsolete due to no corresponding eTICI'
    # MERGED = 'merged into the first study folder due to too short time gap between studies'


class Series_Annotation_Enum(ExtendedEnum):
    OK = 'ok'
    PHASE_CLASSIFICATION = 'selected for phase classification'
    AUTOTICI = 'selected for auto TICI scoring'
    NON_XA = 'not XA'
    UNSUBTRACTED = 'unsubtracted'
    INVERSED = 'inversed'
    DUPLICATED = 'duplicated'


class Patient_Annotation_Enum(ExtendedEnum):
    OK = 'ok'
    UNSPECIFIED = 'unspecified'


class View_Enum(ExtendedEnum):
    AP = 'ap'
    LATERAL = 'lateral'
    BOTH = 'both'


class Pre_Post_EVT_Enum(ExtendedEnum):
    PRE_EVT = 'preEVT'
    POST_EVT = 'postEVT'


class Resolution_Enum(ExtendedEnum):
    LOW = 'low'
    Medium = 'medium'
    HIGH = 'high'


class Hemisphere_Enum(ExtendedEnum):
    LEFT = 'Left hemisphere'
    RIGHT = 'Right hemisphere'
    BOTH = 'both'
    UNSPECIFIED = 'unspecified'


class eTICI_Enum(ExtendedEnum):
    ZERO = '0'
    ONE = '1'
    TWOA = '2A'
    TWOB = '2B'
    TWOC = '2C'
    THREE = '3'
    UNSPECIFIED = 'unspecified'


class mRS_Enum(ExtendedEnum):
    ZERO = '0'
    ONE = '1'
    TWO = '2'
    THREE = '3'
    FOUR = '4'
    FIVE = '5'
    SIX = '6'
    UNSPECIFIED = 'unspecified'


class Occ_Loc_Enum(ExtendedEnum):
    M1 = 'M1'
    M2 = 'M2'
    INTRACRANIAL_ICA = 'Intracranial ICA'
    EXTRACRANIAL_ICA = 'Extracranial ICA'
    A1 = 'A1'
    A2 = 'A2'
    OTHER = 'other: M3/M4/multiple/posterior'
    NONE = 'None'
    UNSPECIFIED = 'unspecified'
