from __future__ import print_function

import argparse
import logging
import os
from pathlib import Path

import SimpleITK as sitk
import ants
import cv2 as cv
import numpy as np


# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
green = (0, 255, 0)
red = (0, 0, 255)


def elastix_align_image_pair(fixed_image, moving_image,
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
            logger.warning("First try for alignment failed. Giving it another try!")
            continue
        break
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    aligned_img = elastixImageFilter.GetResultImage()
    aligned_img = sitk.GetArrayFromImage(aligned_img)

    return aligned_img, transformParameterMap


def elastix_transform_sequence(img_seq, transform_parameter_map):
    transformed_img_seq = img_seq.copy()
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    for i, img in enumerate(img_seq[:]):
        transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(img))
        transformed_img_seq[i] = sitk.GetArrayFromImage(transformixImageFilter.Execute())
    return transformed_img_seq


def elastix_transform_image(img, transform_parameter_map):
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(img))
    transformed_img = sitk.GetArrayFromImage(transformixImageFilter.Execute())
    return transformed_img


def elastix_align_sequence(img_seq):
    aligned_img_seq = img_seq.copy().astype(np.float32)
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    mid_idx = len(img_seq) // 2  # index of the fixed image

    transformParameterMaps = []
    for i in range(mid_idx + 1, img_seq.shape[0]):
        # tic = time.perf_counter()
        aligned_img_seq[i], transformParameterMap = elastix_align_image_pair(sitk.GetImageFromArray(img_seq[i - 1]),
                                                                             sitk.GetImageFromArray(img_seq[i]),
                                                                             transform='rigid')
        # toc = time.perf_counter()
        # print(f"Align time: {toc - tic:0.4f} seconds")
        transformParameterMaps.append(transformParameterMap)

        # aligned_img_seq[i] = img_aligned
        for j in range(len(transformParameterMaps) - 2, -1, -1):
            # tic = time.perf_counter()
            transformixImageFilter.SetTransformParameterMap(transformParameterMaps[j])
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(aligned_img_seq[i]))
            aligned_img_seq[i] = sitk.GetArrayFromImage(transformixImageFilter.Execute())
            # toc = time.perf_counter()
            # print(f"Transformation time: {toc - tic:0.4f} seconds")

    transformParameterMaps = []
    for i in range(mid_idx - 1, -1, -1):
        # tic = time.perf_counter()
        aligned_img_seq[i], transformParameterMap = elastix_align_image_pair(sitk.GetImageFromArray(img_seq[i + 1]),
                                                                             sitk.GetImageFromArray(img_seq[i]),
                                                                             transform='rigid')
        # toc = time.perf_counter()
        # print(f"Align time: {toc - tic:0.4f} seconds")
        transformParameterMaps.append(transformParameterMap)

        for j in range(len(transformParameterMaps) - 2, -1, -1):
            # tic = time.perf_counter()
            transformixImageFilter.SetTransformParameterMap(transformParameterMaps[j])
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(aligned_img_seq[i]))
            aligned_img_seq[i] = sitk.GetArrayFromImage(transformixImageFilter.Execute())
            # toc = time.perf_counter()
            # print(f"Transformation time: {toc - tic:0.4f} seconds")

    return aligned_img_seq


def elastix_group_mc(img_seq, metric=None, n_iteration=None):
    img_seq = img_seq.astype(np.float32)
    vectorOfImages = sitk.VectorOfImage()
    for i in range(img_seq.shape[0]):
        vectorOfImages.push_back(sitk.GetImageFromArray(img_seq[i]))
    image = sitk.JoinSeries(vectorOfImages)

    # Register
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetFixedImage(image)
    elastixImageFilter.SetMovingImage(image)

    if n_iteration is None:
        n_iteration = ['512']
    if metric is None:
        metric = ['AdvancedMattesMutualInformation']
    resolution = 6
    interpolator = ['LinearInterpolator']
    final_resample_interpolator = ['FinalLinearInterpolator']
    final_BSpline_interpolation_order = ['1']
    # transform = ['EulerStackTransform']
    transform = ['AffineLogStackTransform']

    parameterMap = sitk.GetDefaultParameterMap("groupwise", resolution)
    parameterMap['Metric'] = metric
    parameterMap['MaximumNumberOfIterations'] = n_iteration
    parameterMap['ResampleInterpolator'] = final_resample_interpolator
    parameterMap['FinalBSplineInterpolationOrder'] = final_BSpline_interpolation_order
    parameterMap['Transform'] = transform
    parameterMap['Interpolator'] = interpolator
    # parameterMap['MaximumNumberOfSamplingAttempts'] = ['1']

    # default_pixel_value = [str(float(sitk.GetArrayFromImage(moving_image).max()))]
    # parameterMap['DefaultPixelValue'] = default_pixel_value  # default pixel value being 0 or negative seems no diff

    elastixImageFilter.SetParameterMap(parameterMap)

    # elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('groupwise'))
    aligned_image_sequence = sitk.GetArrayFromImage(elastixImageFilter.Execute())
    return aligned_image_sequence


def ants_align_sequence(img_seq):
    # fi = ants.image_read(ants.get_ants_data('files/ch2.nii.gz'))
    sequence_length = img_seq.shape[0]
    img_seq = np.transpose(img_seq, (1, 2, 0))
    mc = ants.motion_correction(ants.from_numpy(img_seq),
                                fixed=ants.from_numpy(img_seq[:, :, int(sequence_length // 2)]),
                                type_of_transform="BOLDRigid", fdOffset=0)
    aligned_img_seq = np.transpose(mc['motion_corrected'].numpy(), (2, 0, 1))
    return aligned_img_seq


def sift_based_align(img1, img2, plot=False):
    MIN_MATCH_COUNT = 10
    """"
    Method for image alignment

    A Homography matrix is calculated by matching the Akaze features between the input image and the reference image.
    """
    sift = cv.xfeatures2d.SIFT_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # plotting the detected key points on top of the image
    img1_with_kp = cv.drawKeypoints(img1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_kp = cv.drawKeypoints(img2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if plot:
        image_comparison1 = cv.hconcat([img1_with_kp, img2_with_kp])
        image_comparison1 = cv.resize(image_comparison1, (800, 450))
        cv.imshow('images with key points: left-img1, right-img2', image_comparison1)
        cv.waitKey(1000)
        cv.destroyAllWindows()

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Flann based matcher
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to select good matched keypoints
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append([m])

    # Warp image if enough matches found
    if len(good_matches) >= MIN_MATCH_COUNT:
        src_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
        dst_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

        # Compute homography
        H, mask = cv.findHomography(dst_kpts, src_kpts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # Warp image
        if H is not None:
            warped_img2 = cv.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))
        else:
            warped_img2 = img2
            matchesMask = None
            H = np.identity(3)
    else:
        logger.error("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None
        warped_img2 = img2
        H = np.identity(3)

    # Plot warp results
    if plot:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=(255, 0, 0),
                           # matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img_matching = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, **draw_params)
        img_matching = cv.resize(img_matching, (800, 450))
        cv.imshow('image match', img_matching)
        cv.waitKey(1000)
        cv.destroyAllWindows()

        # show image comparison between before and after warping
        image_comparison2 = cv.hconcat([img1, warped_img2])
        image_comparison2 = cv.resize(image_comparison2, (800, 450))
        cv.imshow('warp result: left-img1, right-warped_img2', image_comparison2)
        cv.waitKey(1000)
        cv.destroyAllWindows()
    return warped_img2, H


def dsa_sequence_align(img_seq):
    aligned_img_seq = np.zeros(img_seq.shape)
    aligned_img_seq[0] = img_seq[0]
    H_overall = np.identity(3)
    # image_overview = []
    for i in range(1, img_seq.shape[0]):
        _, H = sift_based_align(img_seq[i - 1], img_seq[i])
        H_overall = np.dot(H_overall, H)
        aligned_img_seq[i] = cv.warpPerspective(img_seq[i], H_overall, (img_seq[i].shape[1], img_seq[i].shape[0]),
                                                borderValue=255)

        # image_overview = cv.hconcat(image_overview, cv.vconcat(image_matching, aligned_img_seq[i]))

    # image_overview = cv.resize(image_overview, (1600, 900))
    # cv.imshow('DSA sequence alignment overview', image_overview)
    # cv.waitKey(1000)
    # cv.destroyAllWindows()
    return aligned_img_seq


class OF_based_align:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.p0 = None
        self.use_ransac = True
        # self.input_img_seq = img_seq
        # self.aligned_img_seq = img_seq.copy()
        # self.sift = cv.xfeatures2d.SIFT_create()
        self.feature_params = dict(maxCorners=10000,
                                   qualityLevel=0.01,
                                   minDistance=5,  # 0 based
                                   blockSize=9)  # from github lk_homography.py
        self.lk_params = dict(winSize=(9, 9),
                              maxLevel=6,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))  # from github

    def checkedTrace(self, img0, img1, p0, back_threshold=1.0):
        p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
        p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        status = d < back_threshold

        # Plot matches before frames
        # vis = np.concatenate((img0, img1), axis=1)
        # vis = cv.cvtColor(vis, cv.COLOR_GRAY2RGB)
        # for (x0, y0), (x1, y1), good in zip(p0[:, 0], p1[:, 0].astype(np.int32), status[:]):
        #     if good:
        #         cv.line(vis, (x0, y0), (x1+img0.shape[1], y1), (red, green)[good])
        #     cv.circle(vis, (x1+img0.shape[1], y1), 2, (red, green)[good], -1)
        # cv.imshow('debug.png', vis)
        # cv.waitKey(1000)
        # cv.destroyAllWindows()
        return p1, status

    def image_align(self, img_ref, img_in):
        p0 = cv.goodFeaturesToTrack(img_ref, **self.feature_params)
        H = np.identity(3)
        if p0 is None or len(p0) < 100:
            logger.warning("No good features to track found in the reference image, returning original input image.")
            return img_in, H
        p1, trace_status = self.checkedTrace(img_ref, img_in, p0, back_threshold=5.0)
        p1 = p1[trace_status]
        p0 = p0[trace_status]
        if len(p0) != 0 and len(p1) != 0:
            H, status = cv.estimateAffine2D(p1, p0, method=(0, cv.RANSAC)[self.use_ransac], ransacReprojThreshold=5.0)
        if H is None or len(p0) == 0 or len(p1) == 0:
            logger.warning("No transformation matrix found for the image pair, returning original input image.")
            H = np.identity(3)
            aligned_img = img_in
            vis = self.visualize_points(aligned_img, p0)
        else:
            H = np.append(H, [[0, 0, 1]], axis=0)
            h, w = img_in.shape[:2]
            aligned_img = cv.warpPerspective(img_in, H, (w, h), borderValue=255)
            vis = self.visualize_alignment(aligned_img, img_in, p0, p1, status)
        cv.imwrite('debug.png', vis)
        cv.waitKey(10000)
        cv.destroyAllWindows()
        return aligned_img, H, vis

    def warp_sequence_LK(self, input_img_seq):
        aligned_img_seq = input_img_seq.copy()
        H_overall = np.identity(3)
        for i in range(0, input_img_seq.shape[0]):
            cur_frame = input_img_seq[i].copy()
            if self.p0 is not None:
                p1, trace_status = self.checkedTrace(prev_frame, cur_frame, self.p0, back_threshold=5.0)
                p1 = p1[trace_status]
                self.p0 = self.p0[trace_status]
                status = None
                if len(self.p0) >= 100:
                    H, status = cv.estimateAffine2D(p1, self.p0, method=(0, cv.RANSAC)[self.use_ransac],
                                                    ransacReprojThreshold=5.0)
                    H = np.append(H, [[0, 0, 1]], axis=0)
                    # H, status = cv.findHomography(p1, self.p0, (0, cv.RANSAC)[self.use_ransac], 5.0)
                    # if len(self.p0[status[:, 0]]) >= 10:
                    #     H, status = cv.findHomography(p1[np.ma.make_mask(status[:, 0])], self.p0[np.ma.make_mask(status[:, 0])], cv.LMEDS)
                    if H is not None:
                        H_overall = np.dot(H_overall, H)
                h, w = cur_frame.shape[:2]
                aligned_img_seq[i] = cv.warpPerspective(aligned_img_seq[i], H_overall, (w, h), borderValue=255)

                vis = self.visualize_alignment(aligned_img_seq[i], cur_frame, self.p0, p1, status)
            else:
                # p, _ = self.sift.detectAndCompute(cur_frame, None)
                p = cv.goodFeaturesToTrack(cur_frame, **self.feature_params)
                if p is not None:
                    vis = self.visualize_points(cur_frame, p)

            # show overlay of before and after alignment comparison
            # cv.imshow('lk_homography.png', vis)
            # cv.waitKey(1000)
            # cv.destroyAllWindows()

            # self.frame0 = cur_frame.copy()
            # # Find the keypoints and descriptors with SIFT
            # self.p0, des1 = self.sift.detectAndCompute(cur_frame, None)

            self.p0 = cv.goodFeaturesToTrack(cur_frame, **self.feature_params)
            if self.p0 is not None:
                # self.p1 = self.p0
                # self.img0 = cur_frame
                prev_frame = cur_frame.copy()
        return aligned_img_seq

    @staticmethod
    def visualize_alignment(aligned_img, cur_frame, p0, p1, status):
        vis = cv.cvtColor(cur_frame.copy(), cv.COLOR_GRAY2RGB)
        vis = cv.addWeighted(vis, 0.5, cv.cvtColor(aligned_img.astype(np.uint8), cv.COLOR_GRAY2RGB), 0.5, 0.0)
        for j, ((x0, y0), (x1, y1)) in enumerate(zip(p0[:, 0], p1[:, 0])):
            if status is not None:
                cv.line(vis, (x0, y0), (x1, y1), (red, green)[status[j, 0]])
                cv.circle(vis, (x1, y1), 2, (red, green)[status[j, 0]], -1)
            else:
                cv.line(vis, (x0, y0), (x1, y1), red)
                cv.circle(vis, (x1, y1), 2, red, -1)
        # draw_str(vis, (20, 20), 'track count: %d' % len(p1))
        return vis

    @staticmethod
    def visualize_points(image, points):
        vis = cv.cvtColor(image.copy(), cv.COLOR_GRAY2RGB)
        for x, y in points[:, 0]:
            cv.circle(vis, (x, y), 2, green, -1)
            draw_str(vis, (20, 20), 'feature count: {}'.format(len(points)))
        return vis

    def warp_sequence_FB(self):
        flow = np.zeros((*self.input_img_seq.shape, 2), dtype=np.float32)
        for i in range(1, self.input_img_seq.shape[0]):
            prev_frame = self.input_img_seq[i - 1].copy()
            cur_frame = self.input_img_seq[i].copy()

            flow[i] = cv.calcOpticalFlowFarneback(prev_frame, cur_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            cv.imshow('flow', self.draw_flow(cur_frame, flow[i]))
            cv.waitKey(2000)
            for j in range(i, 0, -1):
                cur_frame = self.warp_flow(cur_frame, flow[j])
            self.aligned_img_seq[i] = cur_frame
        return self.aligned_img_seq

    @staticmethod
    def draw_flow(img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        cv.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    @staticmethod
    def warp_flow(img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        res = cv.remap(img, flow, None, cv.INTER_LINEAR)
        return res


def main(filename, args):
    """"
    Main ica program, is called if not included from a module

    Can be called by other modules when provided a Namespace
    that is similar to the one produced by parse_args
    """
    logger.info("=" * 30)
    logger.info("imr_dsa_ica.main()")

    filename = os.path.join(config.step4_baseline_path_base, filename)
    (_, realfile) = os.path.split(filename)
    (realbase, _) = os.path.splitext(realfile)

    logger.info('Reading image {}'.format(filename))
    try:
        image_orig, origin, spacing = utils.utils.load_itkfile(filename)
    except Exception as error:
        logger.error('Error while reading image {}'.format(filename))
        logger.error('\n' + str(error))
        return

    if image_orig.shape[0] < 12:
        logger.warning("Won't do ica on less then 12 frames ({})".format(image_orig.shape[0]))
        return

    # logger.info('Normalizing image')
    # image_mean = np.mean(image_orig)
    # image_std = np.std(image_orig)
    # logger.info('Image mean = {}, image std = {}'.format(image_mean, image_std))
    # image = (image_orig-image_mean)/image_std
    # cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)

    logger.info('Aligning images in the image sequence')
    aligned_image_seq_of = OF_based_align().warp_sequence_LK(image_orig)
    aligned_image_seq_sift = dsa_sequence_align(image_orig[0:])
    aligned_image_seq_elastix = elastix_align_sequence(image_orig)
    # aligned_image_seq = of_based_align(image_orig[1:]).warp_sequence_FB()
    # aligned_image_seq = aligned_image_seq*image_std+image_mean
    # cv.normalize(aligned_image_seq, aligned_image_seq, 0, 255, cv.NORM_MINMAX)

    dsa_before_alignment = cv.hconcat(image_orig[0:])
    dsa_after_alignment_of = cv.hconcat(aligned_image_seq_of[0:])
    draw_str(dsa_after_alignment_of, (20, 20), 'Optical Flow', font_size=3)
    dsa_after_alignment_sift = cv.hconcat(aligned_image_seq_sift[0:])
    draw_str(dsa_after_alignment_sift, (20, 20), 'SIFT', font_size=3)
    dsa_after_alignment_elastix = cv.hconcat(aligned_image_seq_elastix[0:])
    draw_str(dsa_after_alignment_elastix, (20, 20), 'Elastix affine', font_size=3)
    fig_save_path = os.path.join(os.path.abspath(args.outdir),
                                 get_subject_id(filename) + '_' + get_subject_pre_post(
                                     filename) + '_mc_comparison_' + Path(filename).stem[-7:] + '.png')
    cv.imwrite(fig_save_path, np.concatenate(
        [dsa_before_alignment, dsa_after_alignment_of, dsa_after_alignment_sift, dsa_after_alignment_elastix]))

    '''Save the minip comparison'''
    minip_orig = np.nanmin(image_orig, axis=0)  # Minima along the first axis
    logger.info('Minip original: min = {}, max = {}'.format(minip_orig.min(), minip_orig.max()))
    minip_aligned_of = np.nanmin(aligned_image_seq_of, axis=0)  # Minima along the first axis
    draw_str(minip_aligned_of, (20, 30), 'Optical Flow', font_size=3)
    logger.info('Minip aligned optical flow: min = {}, max = {}'.format(minip_aligned_of.min(), minip_aligned_of.max()))
    minip_aligned_sift = np.nanmin(aligned_image_seq_sift, axis=0)  # Minima along the first axis
    draw_str(minip_aligned_sift, (20, 30), 'SIFT', font_size=3)
    logger.info('Minip aligned SIFT: min = {}, max = {}'.format(minip_aligned_sift.min(), minip_aligned_sift.max()))
    minip_aligned_elastix = np.nanmin(aligned_image_seq_elastix, axis=0)  # Minima along the first axis
    draw_str(minip_aligned_elastix, (20, 30), 'Elastix affine', font_size=3)
    logger.info('Minip aligned Elastix Affine: min = {}, max = {}'.format(minip_aligned_elastix.min(),
                                                                          minip_aligned_elastix.max()))
    # minip_comparison = np.concatenate((minip_orig, minip_aligned, image_orig[0]), axis=1)
    minip_comparison = np.concatenate((minip_orig, minip_aligned_of, minip_aligned_sift, minip_aligned_elastix), axis=1)
    fig_save_path = os.path.join(os.path.abspath(args.outdir),
                                 get_subject_id(filename) + '_' + get_subject_pre_post(
                                     filename) + '_mc_comparison_minip_' + Path(filename).stem[-7:] + '.png')
    cv.imwrite(fig_save_path, minip_comparison)

    logger.info("Done")


def parse_args():
    """
    Argument parser for the main function
    """
    parser = argparse.ArgumentParser(description='Perform SIFT feature based motion correction on DSA sequence')
    parser.add_argument('-i', '--infile', type=readable_file, help='Input image file')
    parser.add_argument('-o', '--outdir', type=readable_dir, default='./output/motion_correction/comparison',
                        help='Output dir for results')
    return parser.parse_args()


if __name__ == '__main__':
    inputfiles = [os.path.join(config.step4_baseline_path_base,
                               'MRCLEAN_R1634/before_001809.718000__9-unknown/1.3.6.1.4.1.40744.9.311811881521209045254178875711621037928-9-920000-r8ip7.dcm')]

    inputfiles = [os.path.join(os.getcwd(), 'files/R0000/output.nii')]
    yaml_cfg = load_yaml('baseline/config.yaml')
    try:
        # inputfiles contains a list of all files to be processed
        # lines starting with a '#' are skipped
        # inputfiles = r'H:\Experiments\imr-dsa-ica\Data\files.txt'
        inputfiles = yaml_cfg['files']
    except ValueError:
        logging.error("Missing files option in configuration file.")
        exit()
    for filename in inputfiles:
        main(filename, parse_args())
