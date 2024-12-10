import argparse

import SimpleITK as sitk
import ants
import cv2 as cv
import numpy as np


def transform_sequence(img_seq, transform_parameter_map):
    transformed_img_seq = img_seq.copy()
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    for i, img in enumerate(img_seq[:]):
        transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(img))
        transformed_img_seq[i] = sitk.GetArrayFromImage(transformixImageFilter.Execute())
    return transformed_img_seq


def transform_image(img, transform_parameter_map):
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(img))
    transformed_img = sitk.GetArrayFromImage(transformixImageFilter.Execute())
    return transformed_img


def register(img_fixed, img_moving):
    """Main function"""
    '''Input'''
    if isinstance(img_fixed, str):
        img_fixed = np.load(img_fixed)
    if isinstance(img_moving, str):
        img_moving = np.load(img_moving)

    '''Blur images'''
    img_fixed = cv.GaussianBlur(img_fixed, (25, 25), 0)
    img_moving = cv.GaussianBlur(img_moving, (25, 25), 0)

    '''Settings'''
    transform = 'affine'
    resolution = 6
    n_iteration = ['512']
    elastix_metric = ['AdvancedMattesMutualInformation']
    ants_metric = 'MattesMutualInformation'
    final_resample_interpolator = ['FinalLinearInterpolator']
    interpolator = ['LinearInterpolator']
    final_BSpline_interpolation_order = ['1']

    parameterMap = sitk.GetDefaultParameterMap(transform, resolution)
    # parameterMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
    parameterMap['Metric'] = elastix_metric
    parameterMap['Interpolator'] = interpolator
    parameterMap['MaximumNumberOfIterations'] = n_iteration
    parameterMap['ResampleInterpolator'] = final_resample_interpolator
    parameterMap['FinalBSplineInterpolationOrder'] = final_BSpline_interpolation_order
    parameterMap['ErodeMask'] = ['false']
    parameterMap['RequiredRatioOfValidSamples'] = ['0.05']

    parameterMap['NumberOfResolutions'] = ['4']
    parameterMap['ImagePyramidSchedule'] = ['8', '8', '4', '4', '2', '2', '1', '1']

    '''Execution'''
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(img_fixed))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(img_moving))

    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.WriteParameterFile(parameterMap, './par0062.txt')
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.Execute()

    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    result_img = np.array(sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()), dtype=np.uint8)

    '''Get mattes mi value'''
    met = ants.create_ants_metric(ants.from_numpy(img_fixed), ants.from_numpy(result_img), metric_type=ants_metric)
    metric_value = abs(met.get_value())

    return metric_value, transformParameterMap


def parse_args():
    """
    Argument parser for the main function
    """
    parser = argparse.ArgumentParser(description='Register a pair of images')
    parser.add_argument('fixed', type=str, help='Input fixed image file path')
    parser.add_argument('moving', type=str, help='Input moving image file path')
    parser.add_argument('mi', type=str, help='Output file path which contains the mattes mi value')
    parser.add_argument('--tdt_mask', type=str, help='Input annotated TDT mask file path on the moving image')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    register(args.fixed, args.moving)
