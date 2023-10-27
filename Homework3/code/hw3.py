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
    for _ in range(len(pairs) * 100):
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
        if len(inliers) == 4: break

    print('min_val', min_val)
    print(len(inliers))
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

def compose(warped, target, scaleFactor):
    composed_img = np.zeros_like(warped)
    height, width, _ = warped.shape
    edge_w = []
    for y in range(height):
        for x in range(width):
            if sum(warped[y][x]) != 0:
                edge_w.append(x)
                break
            if x == width - 1: edge_w.append(x)


    edge_t = []
    for y in range(height):
        for x in range(width):
            if sum(target[y][x]) == 0:
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
        if i == 0: continue
        images.append(cv2.imread(folder_path + file, cv2.IMREAD_COLOR))
    # reference = images[0]
    # keypoints = []
    # descriptors = []
    # for img in images:
    #     gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     sift = cv2.SIFT_create()
    #     kp, des = sift.detectAndCompute(gray,None)
    #     keypoints.append((kp))
    #     descriptors.append((des))
    #     # img=cv2.drawKeypoints(gray,kp,img)
    #     # cv2.imwrite('sift_keypoints.jpg',img)
    #     # break

    target_image = images[0] # queryImage
    for j in range(1, len(images)):
        # Feature matching
        # 1. Given N input images, set one image as a reference
        ref_image = images[j] # trainImage

        # 2. Detect feature points from images and correspondences between pairs of images
        target_kp, target_des = featureExtract(target_image)
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
        H = RANSAC(pairs, 100)
        if H is None: continue
        
        # 4. Warp the images to the reference image
        warped_img = cv2.warpPerspective(ref_image, H, (width * len(images), height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite('warped_img_%d_%d.jpg'%(i,j),warped_img)
        
        # 5. Composite them
        expanded_image = cv2.copyMakeBorder(target_image, 0, 0, 0, width * (len(images) - 1), cv2.BORDER_CONSTANT)
        composed_img = compose(warped_img, expanded_image, 30)
        
        im_mask_zero = np.zeros_like(target_image)
        im_mask_one = np.full_like(target_image, 255)
        im_mask = np.hstack([im_mask_one, im_mask_zero])
        center = (target_image.shape[1]//2, target_image.shape[0]//2)
        im_clone = cv2.seamlessClone(expanded_image, warped_img, im_mask, center, cv2.MIXED_CLONE)
        
        cv2.imwrite('composed_img_%d.jpg'%j,composed_img)
        cv2.imwrite('im_clone%d.jpg'%j,im_clone)
        
        target_image = composed_img
        
        cv2.imwrite('../images/output/6_composed_img.jpg',composed_img)

# files = os.listdir('../images/input/5/')
# files.sort()
# for file in files:
#     from PIL import Image
 
#     image1 = Image.open('../images/input/5/' + file)
#     #이미지의 크기 출력
#     w, h = image1.size
#     # 이미지 자르기 crop함수 이용 ex. crop(left,up, rigth, down)
#     croppedImage=image1.crop((0,300,w,h-300))
    
#     croppedImage.show()
#     print("잘려진 사진 크기 :",croppedImage.size)
    
#     croppedImage.save('../images/input/5/' + file)

panorama('../images/input/6/')