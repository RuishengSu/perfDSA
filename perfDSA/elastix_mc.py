import SimpleITK as sitk
import numpy as np


def elastix_register_image_pair(fixed_image, moving_image,
                             transform='affine',
                             resolution=6,
                             metric=None,
                             n_iteration=None):
    if n_iteration is None:
        n_iteration = ['512']
    if metric is None:
        metric = ['AdvancedMattesMutualInformation']
    final_resample_interpolator = ['FinalLinearInterpolator']
    interpolator = ['LinearInterpolator']
    final_BSpline_interpolation_order = ['1']

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    # elastixImageFilter.SetNumberOfThreads(multiprocessing.cpu_count())

    parameterMap = sitk.GetDefaultParameterMap(transform, resolution)
    parameterMap['Metric'] = metric
    parameterMap['Interpolator'] = interpolator
    parameterMap['MaximumNumberOfIterations'] = n_iteration
    parameterMap['ResampleInterpolator'] = final_resample_interpolator
    parameterMap['FinalBSplineInterpolationOrder'] = final_BSpline_interpolation_order
    # parameterMap['MaximumNumberOfSamplingAttempts'] = ['1']

    # default_pixel_value = [str(float(sitk.GetArrayFromImage(moving_image).max()))]
    # parameterMap['DefaultPixelValue'] = default_pixel_value  # default pixel value being 0 or negative seems no diff

    elastixImageFilter.SetParameterMap(parameterMap)
    # elastixImageFilter.PrintParameterMap()

    for i in range(0, 3):
        try:
            elastixImageFilter.Execute()
        except:
            print("Warning: first try for alignment failed. Giving it another try!")
            continue
        break
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    aligned_img = elastixImageFilter.GetResultImage()
    aligned_img = sitk.GetArrayFromImage(aligned_img)

    return aligned_img, transformParameterMap


def elastix_motion_correction(img_seq):
    aligned_img_seq = img_seq.copy().astype(np.float32)
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    mid_idx = len(img_seq) // 2  # index of the fixed image

    transformParameterMaps = []
    for i in range(mid_idx + 1, img_seq.shape[0]):
        img_aligned, transformParameterMap = elastix_register_image_pair(sitk.GetImageFromArray(img_seq[i - 1]),
                                                                      sitk.GetImageFromArray(img_seq[i]),
                                                                      transform='rigid')
        transformParameterMaps.append(transformParameterMap)

        aligned_img_seq[i] = img_aligned
        for j in range(len(transformParameterMaps) - 2, -1, -1):
            transformixImageFilter.SetTransformParameterMap(transformParameterMaps[j])
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(aligned_img_seq[i]))
            aligned_img_seq[i] = sitk.GetArrayFromImage(transformixImageFilter.Execute())

    transformParameterMaps = []
    for i in range(mid_idx - 1, -1, -1):
        img_aligned, transformParameterMap = elastix_register_image_pair(sitk.GetImageFromArray(img_seq[i + 1]),
                                                                      sitk.GetImageFromArray(img_seq[i]),
                                                                      transform='rigid')
        transformParameterMaps.append(transformParameterMap)

        aligned_img_seq[i] = img_aligned
        for j in range(len(transformParameterMaps) - 2, -1, -1):
            transformixImageFilter.SetTransformParameterMap(transformParameterMaps[j])
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(aligned_img_seq[i]))
            aligned_img_seq[i] = sitk.GetArrayFromImage(transformixImageFilter.Execute())

    return aligned_img_seq