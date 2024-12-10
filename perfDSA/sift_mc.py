import numpy as np
import cv2 as cv


def sift_register_img_pair(img1, img2, plot=False):
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
        print("Error: not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
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
    for i in range(1, img_seq.shape[0]):
        _, H = sift_register_img_pair(img_seq[i - 1], img_seq[i])
        H_overall = np.dot(H_overall, H)
        aligned_img_seq[i] = cv.warpPerspective(img_seq[i], H_overall, (img_seq[i].shape[1], img_seq[i].shape[0]),
                                                borderValue=255)
    return aligned_img_seq
