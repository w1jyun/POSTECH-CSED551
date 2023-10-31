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

def compose(warped, reference, scaleFactor, dir, lu, ld, ru, rd):
    corner_w = [lu[-1], ld[-1], ru[-1], rd[-1]]

    composed_img = reference.copy()
    height, width, _ = reference.shape
    print(corner_w)
    
    def distX(p, up, down):
        d1 = abs(up[1] - p[1])
        d2 = abs(p[1] - down[1])
        r = d1 / (d1+d2)
        h_x = int(r * down[0] + (1-r) * up[0])
        return (abs(h_x - p[0]))

    def distY(p, left, right):
        d1 = abs(left[0] - p[0])
        d2 = abs(p[0] - right[0])
        r = d1 / (d1+d2)
        h_y = int(r * right[1] + (1-r) * left[1])
        return (abs(h_y - p[1]))
    
    w_ = abs(corner_w[0][0]-corner_w[2][0])
    h_ = abs(corner_w[0][1]-corner_w[1][1])

    for y in range(height):
        for x in range(width):
            if sum(warped[y][x]) != 0 and (sum(reference[y][x]) != 0):
                b_w, g_w, r_w = warped[y][x].astype('int')
                b_r, g_r, r_r = reference[y][x].astype('int')
                diff_w_x = distX((x,y), corner_w[0], corner_w[1]) if dir == 1 else distX((x,y), corner_w[2], corner_w[3]) # left, right
                diff_w_y = min(distY((x,y), corner_w[0], corner_w[2]), distY((x,y), corner_w[1], corner_w[3]))
                p_x = (diff_w_x / w_) * scaleFactor if w_ != 0 else 0
                p_y = (diff_w_y / h_) * scaleFactor if h_ != 0 else 0

                p = (p_x * p_y)
                p = max(min(p, 1), 0)

                b = b_w * p + b_r * (1-p)
                g = g_w * p + g_r * (1-p)
                r = r_w * p + r_r * (1-p)
                composed_img[y][x] = [b,g,r]
            else:
                composed_img[y][x] = (warped[y][x] + reference[y][x])

    return composed_img

def panorama(folder_path):
    images = []
    files = os.listdir(folder_path)
    files.sort()

    cnt = 0
    for i, file in enumerate(files):
        if file.split('.')[1] != 'jpeg': continue
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

    lu, ld, ru, rd = [], [], [], []
    lu.append(np.array([pad_w,pad_h,1]).transpose())
    ld.append(np.array([pad_w,pad_h+height,1]).transpose())
    ru.append(np.array([pad_w+width,pad_h,1]).transpose())
    rd.append(np.array([pad_w+width,pad_h+height,1]).transpose())

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

            lu_v = H @ np.array([0,0,1]).transpose()
            ld_v = H @ np.array([0,height,1]).transpose()
            ru_v = H @ np.array([width,0,1]).transpose()
            rd_v = H @ np.array([width,height,1]).transpose()

            lu.append((int(lu_v[0] / lu_v[2]), int(lu_v[1] / lu_v[2])))
            ld.append((int(ld_v[0] / ld_v[2]), int(ld_v[1] / ld_v[2])))
            ru.append((int(ru_v[0] / ru_v[2]), int(ru_v[1] / ru_v[2])))
            rd.append((int(rd_v[0] / rd_v[2]), int(rd_v[1] / rd_v[2])))

            # 5. Composite them
            composed_img = compose(warped_img, reference_img, (width+height)/300, dir, lu, ld, ru, rd)

            cv2.imwrite('../images/output/composed_img_5_%d.jpg'%idx, composed_img)
            reference_img = composed_img
    
    x_min = max(min(p[0] for p in lu), min(p[0] for p in ld))
    x_max = min(max(p[0] for p in ru), max(p[0] for p in rd))
    y_min = max(max(p[1] for p in lu), max(p[1] for p in ru))
    y_max = min(min(p[1] for p in ld), min(p[1] for p in rd))
    result = composed_img[y_min:y_max, x_min:x_max]
    cv2.imwrite('../images/output/result_5.jpg', result)

panorama('../images/input/5/')