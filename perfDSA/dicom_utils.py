import datetime
import logging
import os
import re
from functools import reduce
from pathlib import Path
import glob
import numpy as np
import pydicom
from pydicom import Dataset, FileDataset
from pydicom.errors import InvalidDicomError
from pydicom.tag import Tag
import shutil
import dicom_reader.settings as reader_settings
from dataframe.DSABase import Hemisphere_Enum
# from dataframe.sequence_repository import DSARepository
from dicom_reader import reader as dicom_reader
from dicom_reader.exceptions import *
from dicom_reader.reader import read_file
import cv2 as cv

logger = logging.getLogger(__name__)


def get_sop_uid(ds, key):
    sop = ds.get(key)
    if sop is not None:
        return sop
    sop = ds.get('MediaStorage{}'.format(key))
    if sop is not None:
        return sop
    sop = ds.file_meta.get('MediaStorage{}'.format(key))
    return sop


def compose_filename(sequence_view, hemisphere, series_number, series_id):
    if hemisphere == Hemisphere_Enum.LEFT.value:
        hemisphere = 'left'
    elif hemisphere == Hemisphere_Enum.RIGHT.value:
        hemisphere = 'right'
    return 'SN{}_V{}_H{}_SUID{}'.format(series_number, sequence_view, hemisphere, series_id)


def categorize_images(serie, include_if_no_time_info=True):
    """
    Read all dicom files in a given directory (stop before pixels)
    :type serie: series
    :type stop_before_pixels: bool
    :param stop_before_pixels: bool, whether to stop reading before the pixeldata (handy if we only want header info)
    :param include_if_no_time_info: whether include the dicom files that do not have time info
    :return: List of dicom objects
    """
    singleframe_image_list = []
    multiframe_image_list = []
    excluded_image_list = []
    for img in serie.images:
        ds = img.dicom_dataset
        if (not has_valid_time_info(ds)) and (not include_if_no_time_info):
            logger.error("Excluded image because of missing time info: {}".format(ds.SOPInstanceUID))
            excluded_image_list.append(img)
            continue
        if not is_DSA(ds):
            excluded_image_list.append(img)
            continue
        # if is_secondary(ds):
        #     excluded_image_list.append(img)
        #     continue
        if is_multiframe_dicom(ds):
            multiframe_image_list.append(ds)
        else:
            singleframe_image_list.append(ds)
    return singleframe_image_list, multiframe_image_list, excluded_image_list


def read_dicom_directory(dicom_directory, stop_before_pixels=False, include_if_no_time_info=True):
    """
    Read all dicom files in a given directory (stop before pixels)
    :type stop_before_pixels: bool
    :type dicom_directory: str
    :param stop_before_pixels: Should we stop reading before the pixeldata (handy if we only want header info)
    :param dicom_directory: Directory with dicom data
    :param include_if_no_time_info: whether include the dicom files that do not have time info
    :return: List of dicom objects
    """
    singleframe_ds_list = []
    singleframe_filepath_list = []
    multiframe_filepath_list = []
    for root, _, files in os.walk(dicom_directory):
        for dicom_file in files:
            file_path = os.path.join(root, dicom_file)
            if is_dicom_file(file_path):
                dicom_header = read_file(file_path,
                                         defer_size="1 KB",
                                         stop_before_pixels=stop_before_pixels,
                                         force=reader_settings.pydicom_read_force)
                if not has_valid_time_info(dicom_header):
                    if include_if_no_time_info:
                        pass
                        # logger.warning("Included, but missing time info in dicom header: {}".format(dicom_file))
                    else:
                        logger.error("Excluded because of missing time info in dicom header: {}".format(dicom_file))
                        continue
                if not is_DSA(dicom_header):
                    continue
                if is_multiframe_dicom(dicom_header):
                    multiframe_filepath_list.append(file_path)
                else:
                    singleframe_ds_list.append(dicom_header)
                    singleframe_filepath_list.append(file_path)
    return singleframe_ds_list, multiframe_filepath_list, singleframe_filepath_list


def is_dicom_file(filename):
    """
    Util function to check if file is a dicom file
    the first 128 bytes are preamble
    the next 4 bytes should contain DICM otherwise it is not a dicom

    :param filename: file to check for the DICM header block
    :type filename: str
    :returns: True if it is a dicom file
    """
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    if reader_settings.pydicom_read_force:
        try:
            dicom_headers = pydicom.dcmread(filename, defer_size="1 KB", stop_before_pixels=True, force=True)
            if dicom_headers is not None:
                return True
        except:
            pass
    return False


def is_DSA(dicom_header):
    """
    Function will check if the dicom is of modality: XA
    """
    if 'Modality' not in dicom_header:
        raise MissingModalityError("KEY_MODALITY_NOT_FOUND_IN_HEADER")
    return (dicom_header.Modality == 'XA') and ('1.2.840.10008.5.1.4.1.1.2' not in str(dicom_header.SOPClassUID))


def is_secondary(dicom_header):
    """
    Function will check if the dicom is of modality: XA
    """
    if 'SOPClassUID' not in dicom_header:
        raise MissingModalityError("KEY_SOPClASSUID_NOT_FOUND_IN_HEADER")
    if '1.2.840.10008.5.1.4.1.1.7' in str(dicom_header.SOPClassUID):
        return True
    return False


def is_multiframe_dicom(dicom_header):
    # if ("NumberOfFrames" in dicom_header) and ("FrameIncrementPointer" in dicom_header):
    if ("NumberOfFrames" in dicom_header) and (dicom_header.NumberOfFrames is not None) \
            and (int(dicom_header.NumberOfFrames) > 1):
        return True
    return False


def has_valid_time_info(ds):
    if is_multiframe_dicom(ds) and ('FrameIncrementPointer' in ds):
        if (str(Tag(0x0018, 0x1063)) in str(ds.FrameIncrementPointer)) and \
                'FrameTime' in ds and ds.FrameTime is not None:
            ds.FrameIncrementPointer = Tag(0x0018, 0x1065)
            ds.FrameTimeVector = [0] + [ds.FrameTime] * (ds.NumberOfFrames - 1)
            return True
        elif (ds.FrameIncrementPointer == Tag(0x0018, 0x1065)) and ('FrameTimeVector' in ds) and (ds.FrameTimeVector is not None):
            if 'StartTrim' in ds and 'StopTrim' in ds:
                ds.FrameTimeVector = ds.FrameTimeVector[(ds.StartTrim - 1):ds.StopTrim]
            if len(ds.FrameTimeVector) == ds.NumberOfFrames:
                return True
        return False
    elif ("ContentDate" in ds) and ("ContentTime" in ds):
        try:
            datetime.datetime.strptime(ds.ContentDate, '%Y%m%d').date()
            datetime.datetime.strptime(ds.ContentTime, '%H%M%S.%f').time()
            return True
        except ValueError:
            logger.debug("{}: invalid datetime: {} {}".format(ds.PatientID, ds.ContentDate, ds.ContentTime))
    else:
        logger.debug("{}: missing content date or time in header.".format(ds.PatientID))
    return False


def remove_frames_from_multiframe_dicom(ds, frames_to_remove, out_filepath):
    """
    :param ds: multiframe dicom dataset
    :param frames_to_remove: a list of frame indices to be removed from ds
    :param out_filepath: output file path
    """
    img_sequence = np.dstack(
        [ds.pixel_array[i, :, :] for i in range(ds.pixel_array.shape[0]) if i not in frames_to_remove])
    img_sequence = np.transpose(img_sequence, (2, 0, 1))
    ds.PixelData = img_sequence.tobytes()
    ds.LossyImageCompression = '00'
    if ('FrameIncrementPointer' in ds) and ds.FrameIncrementPointer == Tag(0x0018, 0x1063):
        ds.FrameTimeVector = [0] + [ds.FrameTime] * (ds.NumberOfFrames - 1)

    ds.NumberOfFrames = ds.NumberOfFrames - len(frames_to_remove)
    if ('FrameTimeVector' not in ds) or (ds.FrameTimeVector is None):
        logger.warning("No valid time increments in dicom header!")
    else:
        '''Assign time increments between consecutive frames'''
        ds.FrameIncrementPointer = Tag(0x0018, 0x1065)
        cum_increments = np.cumsum(ds.FrameTimeVector)
        cum_increments = [cum_increments[i] for i in range(len(cum_increments)) if i not in frames_to_remove]
        ds.FrameTimeVector = [0] + [cum_increments[i] - cum_increments[i - 1] for i in range(1, ds.NumberOfFrames)]

    Path(out_filepath).parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(out_filepath, ds, write_like_original=False)


def convert_singleframe_ds_list_to_multiframe_dicom(singleframe_ds_list, out_filepath, convert_if_no_time_info):
    """
    This function collects all single frame dicoms under the input dir and merge them into one multiframe file.

    :param in_dir: directory with the dicom files from a single scan
    :param out_filepath: filepath to the output nifti
    :param convert_if_no_time_info: whether to continue if no valid time info found in single frame dicom files.
    """

    '''Sort single frame ds list'''
    singleframe_ds_list = sort_dicom_frames(singleframe_ds_list)

    time_increments = get_time_increments(singleframe_ds_list)
    if not time_increments:
        if not convert_if_no_time_info:
            logger.error("Aborted conversion as extraction of valid time increments failed!")
            return
        else:
            logger.warning("Converted to multiframe despite missing valid time increments!")

    # sorted_ds_list = sort_dicom_frames(sorted_ds_list)
    Path(out_filepath).parent.mkdir(parents=True, exist_ok=True)

    sequence_data = get_sequence_pixeldata(singleframe_ds_list)
    common_header_items, common_file_meta = get_common_header_items(singleframe_ds_list)

    ds = FileDataset(out_filepath, common_header_items, file_meta=common_file_meta, preamble=b'\0' * 128)

    ds.NumberOfFrames, ds.Columns, ds.Rows = sequence_data.shape[0], sequence_data.shape[2], sequence_data.shape[1]
    ds.PixelData = sequence_data.tobytes()
    ds.LossyImageCompression = '00'
    if 'ImageType' in ds:
        ds.ImageType.append('MERGED')
    else:
        ds.ImageType = ['MERGED']

    '''Assign time increments between consecutive frames'''
    ds.FrameIncrementPointer = Tag(0x0018, 0x1065)
    ds.FrameTimeVector = time_increments
    pydicom.dcmwrite(out_filepath, ds, write_like_original=False)


def sort_dicom_frames(ds_list):
    content_datetime_all_available = True
    acquisition_datetime_all_available = True
    instance_number_all_available = True
    instance_creation_datetime_all_available = True
    for ds in ds_list:
        if ('ContentDate' not in ds) or ('ContentTime' not in ds):
            content_datetime_all_available = False
        if ('AcquisitionDate' not in ds) or ('AcquisitionTime' not in ds):
            acquisition_datetime_all_available = False
        if 'InstanceNumber' not in ds:
            instance_number_all_available = False
        if ('InstanceCreationDate' not in ds) or ('InstanceCreationTime' not in ds):
            instance_creation_datetime_all_available = False

    if all([content_datetime_all_available, acquisition_datetime_all_available,
            instance_creation_datetime_all_available, instance_number_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (
            x.ContentDate, x.ContentTime, x.AcquisitionDate, x.AcquisitionTime, x.InstanceCreationDate,
            x.InstanceCreationTime, x.InstanceNumber))
    elif all([content_datetime_all_available, acquisition_datetime_all_available,
              instance_creation_datetime_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (
            x.ContentDate, x.ContentTime, x.AcquisitionDate, x.AcquisitionTime, x.InstanceCreationDate,
            x.InstanceCreationTime))
    elif all([content_datetime_all_available, acquisition_datetime_all_available, instance_number_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (
            x.ContentDate, x.ContentTime, x.AcquisitionDate, x.AcquisitionTime, x.InstanceNumber))
    elif all([content_datetime_all_available, instance_creation_datetime_all_available, instance_number_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (
            x.ContentDate, x.ContentTime, x.InstanceCreationDate, x.InstanceCreationTime, x.InstanceNumber))
    elif all([acquisition_datetime_all_available, instance_creation_datetime_all_available,
              instance_number_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (
            x.AcquisitionDate, x.AcquisitionTime, x.InstanceCreationDate, x.InstanceCreationTime, x.InstanceNumber))
    elif all([content_datetime_all_available, acquisition_datetime_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (x.ContentDate, x.ContentTime, x.AcquisitionDate))
    elif all([content_datetime_all_available, instance_creation_datetime_all_available]):
        ds_list = sorted(ds_list,
                         key=lambda x: (x.ContentDate, x.ContentTime, x.InstanceCreationDate, x.InstanceCreationTime))
    elif all([acquisition_datetime_all_available, instance_creation_datetime_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (
            x.AcquisitionDate, x.AcquisitionTime, x.InstanceCreationDate, x.InstanceCreationTime))
    elif all([content_datetime_all_available, instance_number_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (x.ContentDate, x.ContentTime, x.InstanceNumber))
    elif all([acquisition_datetime_all_available, instance_number_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (x.AcquisitionDate, x.AcquisitionTime, x.InstanceNumber))
    elif all([instance_creation_datetime_all_available, instance_number_all_available]):
        ds_list = sorted(ds_list, key=lambda x: (x.InstanceCreationDate, x.InstanceCreationTime, x.InstanceNumber))
    elif content_datetime_all_available:
        ds_list = sorted(ds_list, key=lambda x: (x.ContentDate, x.ContentTime))
    elif acquisition_datetime_all_available:
        ds_list = sorted(ds_list, key=lambda x: (x.AcquisitionDate, x.AcquisitionTime))
    elif instance_creation_datetime_all_available:
        ds_list = sorted(ds_list, key=lambda x: (x.InstanceCreationDate, x.InstanceCreationTime))
    elif instance_number_all_available:
        ds_list = sorted(ds_list, key=lambda x: x.InstanceNumber)
    else:
        logger.error("No valid keys in dicom header for sorting files. The file list is not sorted!")
    return ds_list


def get_sequence_pixeldata(sorted_ds_list):
    """
    the slice and intercept calculation can cause the slices to have different dtypes
    we should get the correct dtype that can cover all of them

    :type sorted_ds_list: list of slices
    :param sorted_ds_list: sliced sored in the correct order to create volume
    """
    slices = []
    combined_dtype = None
    for slice_ in sorted_ds_list:
        slice_data = get_slice_pixeldata(slice_)
        if slice_data is None:
            continue
        slice_data = slice_data[np.newaxis, :, :]
        slices.append(slice_data)
        if combined_dtype is None:
            combined_dtype = slice_data.dtype
        else:
            combined_dtype = np.promote_types(combined_dtype, slice_data.dtype)

    # create the new volume with with the correct data
    sequence_data = np.concatenate(slices, axis=0)

    # TODO: check if this transpose is necessary and correct
    # sequence_data = np.transpose(sequence_data, (2, 1, 0))
    return sequence_data


def get_common_header_items(dicom_headers_list):
    """
    This required a dicom list of at least 2.
    """
    # def intersect(d1, d2):
    #     common_ds = Dataset(
    #         dict(filter(lambda elem: (elem[0] in d2) and (d2.get_item(elem[0]).value == elem[1].value), d1.items())))
    #     return common_ds

    def intersect(a, b):
        items = []
        '''Convert all raw data elements to data elements by loop them'''
        for _ in a: pass
        for _ in b: pass

        for i in a.items():
            if (i[0] in b) and (i[0] != Tag(0x7fe0, 0x0010)):
                if b.get_item(i[0]).value == i[1].value:
                    items.append(i)
                else:
                    print('a.{}: {}'.format(i[0], i[1].value))
                    print('b.{}: {}'.format(i[0], b.get_item(i[0]).value))
        return Dataset(dict(items))

    ds1 = reduce(intersect, dicom_headers_list)
    file_meta = dicom_headers_list[0].file_meta
    return ds1, file_meta


def get_time_increments(sorted_ds_list):
    dt_list = []
    for ds in sorted_ds_list:
        try:
            dt = get_frame_datetime(ds)
            dt_list.append(dt)
        except ValueError:
            pass
    if not dt_list:
        return []
    else:
        time_increments = [0.0] + [dt_list[i].timestamp() * 1000 - dt_list[i - 1].timestamp() * 1000 for i in
                                   range(1, len(dt_list))]
        if len(dt_list) != len(sorted_ds_list):
            logger.error("Only {} out of {} frames have timestamps!".format(len(dt_list), len(sorted_ds_list)))
        if np.any([t < 0 for t in time_increments]):
            raise ValueError("Negative time increments: {}".format(time_increments))
        return time_increments


def get_dicom_date_time(ds):
    date = ''
    if 'AcquisitionDate' in ds and ds.AcquisitionDate:
        date = str(ds.AcquisitionDate)
    elif 'SeriesDate' in ds and ds.SeriesDate:
        date = str(ds.SeriesDate)

    timestamp = ''
    if 'AcquisitionTime' in ds and ds.AcquisitionTime:
        timestamp = str(ds.AcquisitionTime)
    elif 'SeriesTime' in ds and ds.SeriesTime:
        timestamp = str(ds.SeriesTime)

    # datetime_str = (date + ' ' + timestamp).split('.')[0]
    # try:
    #     datetime_obj = datetime.strptime(datetime_str, '%Y%m%d %H%M%S')
    #     return datetime_obj.date(), datetime_obj.time()
    # except:
    #     logger.warning(
    #         "Failed parsing datetime string {} into format: %Y%m%d %H%M%S. Returning raw string.".format(datetime_str))
    return date, timestamp


def get_slice_pixeldata(dicom_slice):
    """
    the slice and intercept calculation can cause the slices to have different dtypes
    we should get the correct dtype that can cover all of them

    :type dicom_slice: pydicom object
    :param dicom_slice: slice to get the pixeldata for
    """
    try:
        data = dicom_slice.pixel_array
    except Exception as err:
        logger.error("{}: Dicom pixel array corrupted. Error: {}".format(dicom_slice.PatientID, err))
        return None
    # fix overflow issues for signed data where BitsStored is lower than BitsAllocated and PixelReprentation = 1 (signed)
    # for example a hitachi mri scan can have BitsAllocated 16 but BitsStored is 12 and HighBit 11
    if (dicom_slice.BitsAllocated != dicom_slice.BitsStored) and \
            (dicom_slice.HighBit == dicom_slice.BitsStored - 1) and \
            (dicom_slice.PixelRepresentation == 1):
        if dicom_slice.BitsAllocated == 16:
            data = data.astype(np.int16)  # assert that it is a signed type
        max_value = pow(2, dicom_slice.HighBit) - 1
        invert_value = -1 ^ max_value
        data[data > max_value] = np.bitwise_or(data[data > max_value], invert_value)
    return data


def get_frame_datetime(ds):
    """:param ds: header of a single slice dicom"""
    if ("ContentDate" in ds) and ("ContentTime" in ds):
        try:
            d = datetime.datetime.strptime(ds.ContentDate, '%Y%m%d').date()
            t = datetime.datetime.strptime(ds.ContentTime, '%H%M%S.%f').time()
            dt = datetime.datetime.combine(d, t)
            return dt
        except ValueError:
            raise ValueError("{}: invalid datetime: {} {}".format(ds.PatientID, ds.ContentDate, ds.ContentTime))
    else:
        raise ValueError("{}: missing content time in header".format(ds.PatientID))


def get_patient_id(in_str, prefix='R', suffix=''):
    import re
    try:
        return (re.search('({}[0-9]+{})'.format(prefix, suffix), in_str).group(1)).upper()
    except AttributeError as error:
        logger.error('Error while getting patient id from filepath: {}'.format(in_str))
        logger.error('\n' + str(error))
        return None


def get_pixel_spacing_from_header(ds):
    if 'PixelSpacing' in ds:
        return ds.PixelSpacing
    if ('DistanceSourceToDetector' in ds) and ('DistanceSourceToPatient' in ds) and ('ImagerPixelSpacing' in ds):
        imager_pixel_spacing = np.array([float(s) for s in ds.ImagerPixelSpacing], dtype='float32')
        return imager_pixel_spacing * ds.DistanceSourceToPatient / ds.DistanceSourceToDetector
    return ""


def extract_view_from_filepath(filename):
    try:
        return re.search('V([a-z]*?)_', filename).group(1)
    except AttributeError as error:
        logger.error('Error while extracting view from filepath: {}'.format(filename))
        logger.error('\n' + str(error))
        return 'unknown'


def extract_dicom_view_from_header(ds):
    if isinstance(ds, str):
        ds = dicom_reader.read_file(ds, defer_size="1 KB", stop_before_pixels=True, force=True)

    view = 'unknown'
    if ('PatientOrientation' in ds) and ds.PatientOrientation and \
            (len(ds.PatientOrientation) == 2) and (ds.PatientOrientation[1] == 'F'):
        if ds.PatientOrientation[0] in ['L', 'R']:
            view = 'ap'
        elif ds.PatientOrientation[0] in ['A', 'P']:
            view = 'lateral'
    if ('PositionerPrimaryAngle' in ds) and (ds.PositionerPrimaryAngle is not None):
        if abs(float(ds.PositionerPrimaryAngle)) > 45:
            if view == 'ap':
                logger.warning("Conflicting view: Patient Orientation = {} but PositionerPrimaryAngle = {}. "
                               "Using PositionerPrimaryAngle. Patient: {}, file: {}"
                               "".format(ds.PatientOrientation, ds.PositionerPrimaryAngle, ds.PatientID, ds.SeriesInstanceUID))
            view = 'lateral'
        else:
            if view == 'lateral':
                logger.warning("Conflicting view: Patient Orientation = {} but PositionerPrimaryAngle = {}. "
                               "Using PositionerPrimaryAngle. Patient: {}, file: {}"
                               "".format(ds.PatientOrientation, ds.PositionerPrimaryAngle, ds.PatientID, ds.SeriesInstanceUID))
            view = 'ap'
    return view


def extract_series_number_from_filepath(filename):
    try:
        return re.search('SN([0-9a-z]*?)_', filename).group(1)
    except AttributeError as error:
        logger.error('Error while extracting series number from filepath: {}'.format(filename))
        logger.error('\n' + str(error))
        return 'unknown'


def remove_all_empty_dirs(base_dir):
    dir_paths = [f for f in glob.glob(base_dir + "/**/", recursive=True)]
    dir_paths.reverse()
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            continue

        file_paths = [f for f in glob.glob(os.path.join(dir_path + "**/*"), recursive=True) if
                      os.path.isfile(f)]
        # number_of_dicoms = [f for f in glob.glob(dir_path + "/**/*.dcm", recursive=True)]
        if not file_paths:
            logger.warning('Removing empty dir: {}'.format(dir_path))
            # send2trash(dir_path)
            shutil.rmtree(dir_path)


def draw_str(dst, target, s, font_size=10, thickness=2, color=(255, 255, 255)):
    x, y = target
    cv.putText(dst, s, (x + thickness // 2, y + thickness // 2), cv.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0),
               thickness=thickness, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, font_size, color, thickness=thickness, lineType=cv.LINE_AA)


def add_text_on_image_top(img, txt):
    label_image = cv.putText(img=np.zeros((100, *img.shape[1:])), text=txt, org=(img.shape[1] // 2 - 50, 70),
                             fontFace=3, fontScale=3, color=255, thickness=5)
    img_with_text = np.concatenate([label_image, img])
    return img_with_text


def display_image_file(img_path):
    vis = cv.imread(img_path, 0)
    windowName = "image"
    # cv.namedWindow("sequence", cv.WINDOW_NORMAL)
    cv.namedWindow(windowName, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(windowName, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow(windowName, vis)
    cv.waitKey()
    cv.destroyAllWindows()


def visualize_sequence(ds_seq, save=False, save_path='', show=True, nr_images_per_row=10):
    frames = []
    pixel_array = ds_seq.pixel_array
    pixel_array = np.transpose(pixel_array, (1, 2, 0))
    pixel_array = cv.normalize(pixel_array, None, 0, 255, cv.NORM_MINMAX)
    if pixel_array.ndim == 2:
        vis = pixel_array
    else:
        for idx in range(pixel_array.shape[2]):
            frame = pixel_array[:, :, idx].copy()
            frame = add_text_on_image_top(frame, str(idx))
            frames.append(frame)
        if nr_images_per_row is None:
            nr_images_per_row = pixel_array.shape[2]
        vis = []
        for i in range(int(np.ceil(pixel_array.shape[2] / nr_images_per_row))):
            vis_row = np.concatenate(
                frames[(nr_images_per_row * i):min((nr_images_per_row * i + nr_images_per_row), pixel_array.shape[2])],
                axis=1)
            if vis_row.shape[1] < nr_images_per_row * pixel_array.shape[1]:
                vis_ext_shape = (vis_row.shape[0], nr_images_per_row * pixel_array.shape[1] - vis_row.shape[1])
                vis_row = np.concatenate([vis_row, np.ones(vis_ext_shape, dtype=np.uint8) * 255], axis=1)
            vis.append(vis_row)
        vis = np.concatenate(vis)
        # cv.imwrite('debug.png', vis)

    if save:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(save_path, vis)
    if show:
        windowName = "image"
        # cv.namedWindow("sequence", cv.WINDOW_NORMAL)
        cv.namedWindow(windowName, cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty(windowName, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow(windowName, vis)
        cv.waitKey()
        cv.destroyAllWindows()
    return vis


def visualize_sequence_minip(ds_seq, save=False, save_path='', show=True):
    pixel_array = ds_seq.pixel_array
    pixel_array = np.transpose(pixel_array, (1, 2, 0))
    pixel_array = cv.normalize(pixel_array, None, 0, 255, cv.NORM_MINMAX)
    if pixel_array.ndim == 2:
        vis = pixel_array
    else:
        vis = np.min(pixel_array, 2)

    if save:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(save_path, vis)
    if show:
        windowName = "image"
        # cv.namedWindow("sequence", cv.WINDOW_NORMAL)
        cv.namedWindow(windowName, cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty(windowName, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow(windowName, vis)
        cv.waitKey()
        cv.destroyAllWindows()
    return vis


def is_valid_dicom(dicom_path):
    try:
        ds = pydicom.read_file(dicom_path, stop_before_pixels=True)
    except IOError:
        return False, None, dicom_path
    except InvalidDicomError:
        return False, None, dicom_path
    '''If the file is a valid dicom file without extension, add .dcm extension to the file name'''
    if not dicom_path.endswith('.dcm'):
        logger.info("Renaming dicom file without extension: {}".format(dicom_path))
        os.rename(dicom_path, dicom_path + '.dcm')
        return True, ds, dicom_path + '.dcm'
    return True, ds, dicom_path
