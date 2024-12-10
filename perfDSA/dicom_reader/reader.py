import os
import subprocess
import tempfile

import pydicom

from dicom_reader.exceptions import ConversionError
from dicom_reader import settings


def read_file(dicom_file, defer_size=None, stop_before_pixels=False, force=False):
    if _is_compressed(dicom_file, force):
        # https://github.com/icometrix/dicom2nifti/issues/46 thanks to C-nit
        try:
            with tempfile.NamedTemporaryFile(delete=False) as fp:
                fp.close()
                _decompress_dicom(dicom_file, output_file=fp.name)

            return pydicom.dcmread(fp.name,
                                   defer_size=None,  # We can't defer
                                   stop_before_pixels=stop_before_pixels,
                                   force=force)
        finally:
            os.remove(fp.name)

    dicom_header = pydicom.dcmread(dicom_file,
                                   defer_size=defer_size,
                                   stop_before_pixels=stop_before_pixels,
                                   force=force)
    return dicom_header


def _is_compressed(dicom_file, force=False):
    """
    Check if dicoms are compressed or not
    """
    header = pydicom.dcmread(dicom_file, defer_size="1 KB", stop_before_pixels=True, force=force)

    uncompressed_types = ["1.2.840.10008.1.2",
                          "1.2.840.10008.1.2.1",
                          "1.2.840.10008.1.2.1.99",
                          "1.2.840.10008.1.2.2"]

    if 'TransferSyntaxUID' in header.file_meta and header.file_meta.TransferSyntaxUID in uncompressed_types:
        return False
    return True


def _decompress_dicom(dicom_file, output_file):
    """
    This function can be used to convert a jpeg compressed image to an uncompressed one for further conversion

    :param input_file: single dicom file to decompress
    """
    gdcmconv_executable = _get_gdcmconv()

    subprocess.check_output([gdcmconv_executable, '-w', dicom_file, output_file])


def _get_gdcmconv():
    """
    Get the full path to gdcmconv.
    If not found raise error
    """
    gdcmconv_executable = settings.gdcmconv_path
    if gdcmconv_executable is None:
        gdcmconv_executable = _which('gdcmconv')
    if gdcmconv_executable is None:
        gdcmconv_executable = _which('gdcmconv.exe')

    if gdcmconv_executable is None:
        raise ConversionError('GDCMCONV_NOT_FOUND')

    return gdcmconv_executable


def _which(program):
    import os

    def is_exe(executable_file):
        return os.path.isfile(executable_file) and os.access(executable_file, os.X_OK)

    file_path, file_name = os.path.split(program)
    if file_path:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None