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
    # pair_ref = pairs[random.randint(0, len(pairs)-1)]
    # find largest inlier pair set
    inliers = []
    for _ in range(len(pairs) * 100):
        pair_set = [pairs[random.randint(0, len(pairs)-1)] for _ in range(4)]
        inliers_tmp = []
        H = homography(pair_set)
        for pair in pair_set:
            src, dst = pair
            src_v = np.array([src[0], src[1], 1])
            dst_v = np.array([dst[0], dst[1], 1])
            if SSD(dst_v, H @ src_v) < threshold:
                inliers_tmp.append(H)

        if len(inliers_tmp) > len(inliers):
            inliers = copy.deepcopy(inliers_tmp)
        if len(inliers) == 4: break

    print('inliers', len(inliers))
    if len(inliers) == 0:
        return None
    # calculate average H
    avg_h = np.zeros_like(inliers[0])
    for h in inliers:
        avg_h += h
    avg_h /= len(inliers)
    # np.linalg.lstsq(np.vstack(inliers).T)
    return avg_h

def featureExtract(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray,None)
    return kp, des

def compose(warped, target, scaleFactor, dir):
    composed_img = target.copy()
    height, width, _ = target.shape

    # find edge
    edge_w = []
    edge_t = []

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
                is_edge = (sum(target[y][x]) == 0)
                if is_edge:
                    edge_t.append(x)
                    break
                if x == width - 1: edge_t.append(x)
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
                is_edge = (sum(target[y][x]) == 0)
                if is_edge:
                    edge_t.append(x)
                    break
                if x == width - 1: edge_t.append(x)
            
    for y in range(height):
        for x in range(width):
            if sum(warped[y][x]) != 0 and sum(target[y][x]) != 0:
                b_a, g_a, r_a = warped[y][x].astype('int')
                b_b, g_b, r_b = target[y][x].astype('int')
                diff1 = abs(edge_w[y] - x)
                diff2 = abs(edge_t[y] - x)

                p = (diff1 / (diff1 + diff2)) * scaleFactor
                p = min(max(p, 0), 1)
                b = b_a * p + b_b * (1-p)
                g = g_a * p + g_b * (1-p)
                r = r_a * p + r_b * (1-p)
                composed_img[y][x] = [b, g, r]
            else:
                composed_img[y][x] = (warped[y][x] + target[y][x])

    return composed_img

def panorama(folder_path):
    images = []
    files = os.listdir(folder_path)
    files.sort()
    for i, file in enumerate(files):
        images.append(cv2.imread(folder_path + file, cv2.IMREAD_COLOR))

    mid = math.ceil(len(files) / 2)
    target_image = images[mid] # queryImage
    height, width, _ = target_image.shape
    pad_h = height // 4
    pad_w = (len(files) * width) // 3
    target_image = cv2.copyMakeBorder(target_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

    for i in range(1, mid+1):
        for j in [-1, 1]:
            idx = mid + i * j
            if idx < 0 or idx >= len(files): continue
            # Feature matching
            # 1. Given N input images, set one image as a reference
            ref_image = images[idx] # trainImage

            # 2. Detect feature points from images and correspondences between pairs of images
            target_kp, target_des = featureExtract(target_image)
            ref_kp, ref_des = featureExtract(ref_image)

            pairs = []
            matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            matches = matcher.match(ref_des, target_des)
            matches = sorted(matches, key = lambda x:x.distance)
            
            for match in matches:
                idx_i = match.queryIdx
                idx_j = match.trainIdx
                pairs.append((ref_kp[idx_i].pt, target_kp[idx_j].pt))
            
            # 3. Estimate the homographies between images using RANSAC
            H = RANSAC(pairs, 100)
            if H is None: continue

            # 4. Warp the images to the reference image
            warped_img = cv2.warpPerspective(ref_image, H, (target_image.shape[1], target_image.shape[0]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # 5. Composite them
            composed_img = compose(warped_img, target_image, width / 50, j)
            
            cv2.imwrite('composed_img_%d.jpg'%i,composed_img)
            target_image = composed_img
            cv2.imwrite('../images/output/2_1_composed_img.jpg',composed_img)

panorama('../images/input/2/')