import sys
import cv2
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import exifread

# White balance
def white_balance(img, cfa_type):
    H, W = img.shape
    b = []
    g = []
    r = []
    # RGGB / GBRG / GRBG / RGGB
    for h in range(H):
        for w in range(W):
            idx = (h % 2) * 2 + w % 2
            if cfa_type[idx] == 'R':
                r.append(img[h][w])
            elif cfa_type[idx] == 'G':
                g.append(img[h][w])
            elif cfa_type[idx] == 'B':
                b.append(img[h][w])

    b_avg = np.mean(b)
    g_avg = np.mean(g)
    r_avg = np.mean(r)

    r_coeff = g_avg / r_avg
    b_coeff = g_avg / b_avg
    print(r_coeff, b_coeff)
    balanced_img = img.copy()
    for h in range(H):
        for w in range(W):
            idx = (h % 2) * 2 + w % 2
            if cfa_type[idx] == 'R':
                balanced_img[h][w] = min(balanced_img[h][w] * r_coeff, 1.0)
            elif cfa_type[idx] == 'B':
                balanced_img[h][w] = balanced_img[h][w] * b_coeff

    return balanced_img

# CFA interpolation
def interpolation(img, cfa_type):
    H, W = img.shape
    interpolated_img = np.zeros((H,W,3))

    def isOut(h,w):
        return h < 0 or h >= H or w < 0 or w >= W
    
    def get_w(h,w):
        if w == 0:
            return img[h][w+1]
        elif w == W-1:
            return img[h][w-1]
        else:
            return np.mean([img[h][w-1], img[h][w+1]])

    def get_h(h,w):
        if h == 0:
            return img[h+1][w]
        elif h == H-1:
            return img[h-1][w]
        else:
            return np.mean([img[h+1][w], img[h-1][w]])

    def get_x(h,w):
        values = []
        if not isOut(h-1,w-1): values.append(img[h-1][w-1])
        if not isOut(h-1,w+1): values.append(img[h-1][w+1])
        if not isOut(h+1,w-1): values.append(img[h+1][w-1])
        if not isOut(h+1,w+1): values.append(img[h+1][w+1])
        return np.mean(values)
    
    def get_g(h,w):
        values = []
        if not isOut(h-1,w): values.append(img[h-1][w])
        if not isOut(h+1,w): values.append(img[h+1][w])
        if not isOut(h,w-1): values.append(img[h][w-1])
        if not isOut(h,w+1): values.append(img[h][w+1])
        return np.mean(values)
    

    for h in range(H):
        for w in range(W):
            idx = (h % 2) * 2 + w % 2
            if cfa_type[idx] == 'R':
                r = img[h][w]
                g = get_g(h,w)
                b = get_x(h,w)
                interpolated_img[h][w] = [b,g,r]
            elif cfa_type[idx] == 'G':
                r = get_w(h,w)
                g = img[h][w]
                b = get_h(h,w)
                interpolated_img[h][w] = [b,g,r]
            elif cfa_type[idx] == 'B':
                r = get_x(h,w)
                g = get_g(h,w)
                b = img[h][w]
                interpolated_img[h][w] = [b,g,r]

    return interpolated_img

# Gamma correction
def gamma_correction(img):
    result_img = img ** (1 / 2.2)
    return result_img

if __name__ == '__main__':
    tiff_path = sys.argv[1]
    # 이미지 파일 열기
    tags = exifread.process_file(open(tiff_path, 'rb'))
    # 태그 출력
    metadata = {f"{tag}" : f"{value}" for tag, value in tags.items()}
    try:
        # BGGR, RGBG, GRGB, RGGB
        # GRBG, RGGB, BGGR, RGGB
        isVertical = metadata['Image ImageWidth'] < metadata['Image ImageLength']
        if metadata['Image Model'] == 'SM-G935L': # galaxy S7 edge
            cfa_type = 'BGGR' if isVertical else 'GRBG'
        elif metadata['Image Model'] == 'SM-G991N': # galaxy S21
            cfa_type = 'GRBG' if isVertical else 'BGGR'
        else:
            raise Exception('Unknown Model')   
    except Exception as e:
        print('Unknown Model', e)

    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    img = img / (pow(2,16) - 1)
    balanced_img = white_balance(img, cfa_type)
    interpolated_img = interpolation(balanced_img, cfa_type)
    cv2.imwrite('interpolated_img.jpg', (interpolated_img * 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    result_img = gamma_correction(interpolated_img)
    filename = tiff_path.split('.')[0]
    result_img *= 255
    cv2.imwrite(f'{filename}.jpg', result_img.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    