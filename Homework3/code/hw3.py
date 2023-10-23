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
    min_val = float('inf')
    for _ in range(len(pairs)):
        pair_set = [pairs[random.randint(0, len(pairs)-1)] for _ in range(4)]
        inliers_tmp = []
        H = homography(pair_set)
        for pair in pair_set:
            src, dst = pair
            src_v = np.array([src[0], src[1], 1])
            dst_v = np.array([dst[0], dst[1], 1])
            min_val = min(SSD(dst_v, H @ src_v), min_val)
            if SSD(dst_v, H @ src_v) < threshold:
                inliers_tmp.append(H)
        if len(inliers_tmp) > len(inliers):
            inliers = copy.deepcopy(inliers_tmp)

    print('min_val', min_val)

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

def compose(warped, target):
    composed_img = np.zeros_like(warped)
    height, width, _ = warped.shape
    minX = float('inf')
    minValues = []
    for y in range(height):
        for x in range(width):
            if sum(warped[y][x]) != 0:
                minX = min(x, minX)
        minValues.append(minX)

    for y in range(height):
        for x in range(width):
            if sum(warped[y][x]) != 0 and sum(target[y][x]) != 0:
                b_a, g_a, r_a = warped[y][x].astype('int')
                b_b, g_b, r_b = target[y][x].astype('int')
                p = (abs(minValues[y] - x)) / (abs(minValues[y] - x) + abs(width // 2 - x))
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
    print(files)
    for file in files:
        images.append(cv2.imread(folder_path + file, cv2.IMREAD_COLOR))
        
    # 1. Given N input images, set one image as a reference
    # reference = images[0]
    # keypoints = []
    # descriptors = []
    # # 2. Detect feature points from images and correspondences between pairs of images
    # for img in images:
    #     gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     sift = cv2.SIFT_create()
    #     kp, des = sift.detectAndCompute(gray,None)
    #     keypoints.append((kp))
    #     descriptors.append((des))
    #     # img=cv2.drawKeypoints(gray,kp,img)
    #     # cv2.imwrite('sift_keypoints.jpg',img)
    #     # break

    total_H = np.identity(3)
    for i in range(len(images)-1):
          j = i+1
          # Feature matching
          target_image = images[i] # queryImage
          target_kp, target_des = featureExtract(target_image)
          ref_image = images[j] # trainImage
          ref_kp, ref_des = featureExtract(ref_image)
          height, width, _ = ref_image.shape
          pairs = []
          matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
          matches = matcher.match(ref_des, target_des)
          matches = sorted(matches, key = lambda x:x.distance)
        #   res = cv2.drawMatches(ref_image, ref_kp, target_image,target_kp, matches[:500], None)
        #   cv2.imwrite('matching_%d_%d.jpg'%(i,j), res)

          for match in matches:
              idx_i = match.queryIdx
              idx_j = match.trainIdx
              pairs.append((ref_kp[idx_i].pt, target_kp[idx_j].pt))
          # 3. Estimate the homographies between images using RANSAC
          H = RANSAC(pairs, 50)
          if H is None: continue
          total_H = H @ total_H
          # 4. Warp the images to the reference image
          warped_img = np.zeros((width * 2, height * 2, 3))

        #   for y in range(height * 2):
        #       for x in range(width * 2):
        #           position = np.array([[x, y, 1]]).transpose()
        #           origin_position = np.squeeze(np.linalg.inv(H) @ position)
        #           origin_x = int(origin_position[0] / origin_position[2])
        #           origin_y = int(origin_position[1] / origin_position[2])
        #           if not (origin_x < 0 or origin_x >= width or origin_y < 0 or origin_y >= height):
        #               _y = y + offset_h
        #               _x = x + offset_w
        #               warped_img[_y][_x] = ref_image[origin_y][origin_x]
        
          # for y in range(height):
          #     for x in range(width):
          #         position = np.array([[x, y, 1]]).transpose()
          #         new_position = np.squeeze(H @ position)
          #         new_x = (new_position[0] / new_position[2])
          #         new_x = int(new_x) + offset_w
          #         new_y = (new_position[1] / new_position[2])
          #         new_y = int(new_y) + offset_h
          #         if not (new_x < 0 or new_x >= width * 2 or new_y < 0 or new_y >= height * 2):
          #             warped_img[new_y][new_x] = ref_image[y][x]

          warped_img = cv2.warpPerspective(ref_image, H, (width * 2, height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
          cv2.imwrite('warped_img_%d_%d.jpg'%(i,j),warped_img)

          # 5. Composite them
          expanded_image = cv2.copyMakeBorder(target_image, 0, 0, 0, width, cv2.BORDER_CONSTANT)
          composed_img = compose(warped_img, expanded_image)
          
          im_mask_zero = np.zeros_like(target_image)
          im_mask_one = np.full_like(target_image, 255)
          im_mask = np.hstack([im_mask_one, im_mask_zero])
          center = (target_image.shape[1]//2, target_image.shape[0]//2)

          im_clone = cv2.seamlessClone(expanded_image, warped_img, im_mask, center, cv2.MIXED_CLONE)

          cv2.imwrite('ref_img.jpg',images[i])
          cv2.imwrite('target_img.jpg',images[j])
          cv2.imwrite('composed_img_%d_%d.jpg'%(i,j),composed_img)
          cv2.imwrite('im_clone%d_%d.jpg'%(i,j),im_clone)


panorama('../images/input/2/')