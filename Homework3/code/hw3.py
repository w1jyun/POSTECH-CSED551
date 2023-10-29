# PANORAMA
# 1. Given N input images, set one image as a reference
# 2. Detect feature points from images and correspondences between pairs of images
# 3. Estimate the homographies between images using RANSAC
# 4. Warp the images to the reference image
# 5. Composite them

import cv2
import os
import numpy as np
import random
import copy
import math
from matplotlib import pyplot as plt

sift = cv2.SIFT_create()

def homography(pair_set):
    src_points = np.float32([pair[0] for pair in pair_set]).reshape(-1,1,2)
    dst_points = np.float32([pair[1] for pair in pair_set]).reshape(-1,1,2)

    return cv2.getPerspectiveTransform(src_points, dst_points)

#SumofSquaredDistances
def SSD(pair1, pair2):
    return (sum((pair1 - pair2) * (pair1 - pair2)))

def RANSAC(pairs, threshold):
    # Randomly select one pair in ref
    np.printoptions(precision=2, suppress=True, threshold=5)
    # find largest inlier pair set
    inliers = []
    for _ in range(len(pairs)):
        pair_set = [pairs[random.randint(0, len(pairs)-1)] for _ in range(4)]
        inliers_tmp = []
        H = homography(pair_set)
        for pair in pairs:
            src, dst = pair
            src_v = np.array([src[0], src[1], 1])
            dst_v = np.array([dst[0], dst[1], 1])
            if SSD(dst_v, H @ src_v) < threshold:
                inliers_tmp.append(pair)

        if len(inliers_tmp) > len(inliers):
            inliers = copy.deepcopy(inliers_tmp)

    # calculate average H
    S = np.array([[src[0], src[1], 1] for src, _ in inliers])
    D = np.array([[dst[0], dst[1], 1] for _, dst in inliers])

    return np.linalg.lstsq(S, D, rcond=None)[0].transpose()

def featureExtract(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray,None)
    return kp, des

def compose(warped, reference, scaleFactor, dir):
    composed_img = reference.copy()
    height, width, _ = reference.shape

    # find edge
    edge_w = []
    edge_r = []

    w_mid = width // 2
    if dir == 1:
        # warped image, left edge
        for y in range(height):
            for x in range(width):
                is_edge = (sum(warped[y][x]) != 0)
                if is_edge:
                    edge_w.append(x)
                    break
                if x == width - 1: edge_w.append(x)

        # target image, right edge
        for y in range(height):
            for x in range(w_mid, width):
                is_edge = (sum(reference[y][x]) == 0)
                if is_edge:
                    edge_r.append(x)
                    break
                if x == width - 1: edge_r.append(x)
    else:
        # warped image, right edge
        for y in range(height):
            cnt = 0
            for x in range(width):
                is_black = (sum(warped[y][x]) == 0)
                if cnt == 0 and not is_black:
                    cnt += 1
                if cnt == 1 and is_black:
                    edge_w.append(x)
                    break
                if x == width - 1: edge_w.append(x)

        # target image, left edge
        for y in range(height):
            for x in range(w_mid):
                is_edge = (sum(reference[y][x]) == 0)
                if is_edge:
                    edge_r.append(x)
                    break
                if x == width - 1: edge_r.append(x)
            
    for y in range(height):
        for x in range(width):
            if sum(warped[y][x]) != 0 and (sum(reference[y][x]) != 0):
                b_a, g_a, r_a = warped[y][x].astype('int')
                b_b, g_b, r_b = reference[y][x].astype('int')
                diff1 = abs(edge_w[y] - x)
                diff2 = abs(edge_r[y] - x)

                p = (diff1 / (diff1 + diff2)) * scaleFactor
                p = min(max(p, 0), 1)
                b = b_a * p + b_b * (1-p)
                g = g_a * p + g_b * (1-p)
                r = r_a * p + r_b * (1-p)
                composed_img[y][x] = [b, g, r]
            else:
                composed_img[y][x] = (warped[y][x] + reference[y][x])

    return composed_img


def postprocessing(img, lu, ld, ru, rd):
    lu = (int(lu[0] / lu[2]), int(lu[1] / lu[2]))
    ld = (int(ld[0] / ld[2]), int(ld[1] / ld[2]))
    ru = (int(ru[0] / ru[2]), int(ru[1] / ru[2]))
    rd = (int(rd[0] / rd[2]), int(rd[1] / rd[2]))

    img_height = ld[1] - lu[1] 
    img_width = ru[0] - lu[0] 
    src_points = np.float32([lu, ld, ru, rd])
    dst_points = np.float32([(0,0),(0,img_height),(img_width,0),(img_width,img_height)])
    T = cv2.getPerspectiveTransform(src_points, dst_points)
    dst = cv2.warpPerspective(img, T, (img_width,img_height))
    return dst

def panorama(folder_path):
    images = []
    files = os.listdir(folder_path)
    files.sort()

    cnt = 0
    for i, file in enumerate(files):
        if file.split('.')[1] != 'jpg': continue
        cnt += 1
        images.append(cv2.imread(folder_path + file, cv2.IMREAD_COLOR))

    mid = math.ceil(cnt / 2) - 1
    # 1. Given N input images, set one image as a reference
    reference_img = images[mid]
    height, width, _ = reference_img.shape
    pad_h = height // 4
    pad_w = (cnt * width) // 3
    reference_img = cv2.copyMakeBorder(reference_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)
    composed_img = None
    lu, ld, ru, rd = None, None, None, None
    for i in range(1, mid+2):
        for dir in [-1, 1]:
            idx = mid + i * dir
            print('idx', mid, dir, idx)
            if idx < 0 or idx >= cnt: continue
            # Feature matching
            target_img = images[idx]

            # 2. Detect feature points from images and correspondences between pairs of images
            reference_kp, reference_des = featureExtract(reference_img)
            target_kp, target_des = featureExtract(target_img)

            pairs = []
            matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            matches = matcher.match(target_des, reference_des)
            matches = sorted(matches, key = lambda x:x.distance)
            
            for match in matches:
                idx_i = match.queryIdx
                idx_j = match.trainIdx
                pairs.append((target_kp[idx_i].pt, reference_kp[idx_j].pt))
            
            # 3. Estimate the homographies between images using RANSAC
            H = RANSAC(pairs, 10)
            if H is None: continue

            # 4. Warp the images to the reference image
            warped_img = cv2.warpPerspective(target_img, H, (reference_img.shape[1], reference_img.shape[0]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # 5. Composite them
            if dir == -1:
                lu = H @ np.array([0,0,1]).transpose()
                ld = H @ np.array([0,height,1]).transpose()

            else:
                ru = H @ np.array([width,0,1]).transpose()
                rd = H @ np.array([width,height,1]).transpose()

            composed_img = compose(warped_img, reference_img, width / 50, dir)
            cv2.imwrite('../images/output/composed_img_4_%d.jpg'%idx,composed_img)
            reference_img = composed_img
    
    result = postprocessing(composed_img, lu, ld, ru, rd)
    cv2.imwrite('../images/output/result_4.jpg', result)

panorama('../images/input/4/')