import sys
import cv2
import numpy as np
import exifread

# White balance
def white_balance(img, cfa_type):
    H, W = img.shape
    b = []
    g = []
    r = []
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
    balanced_img = img.copy()
    for h in range(H):
        for w in range(W):
            idx = (h % 2) * 2 + w % 2
            if cfa_type[idx] == 'R':
                balanced_img[h][w] = balanced_img[h][w] * r_coeff
            elif cfa_type[idx] == 'B':
                balanced_img[h][w] = balanced_img[h][w] * b_coeff
    balanced_img = np.clip(balanced_img, 0, 1)
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
                w_val = get_w(h,w)
                h_val = get_h(h,w)
                if idx % 2 == 0:
                    b = w_val if cfa_type[idx + 1] == 'B' else h_val
                    r = h_val if cfa_type[idx + 1] == 'B' else w_val     
                else:
                    b = w_val if cfa_type[idx - 1] == 'B' else h_val
                    r = h_val if cfa_type[idx - 1] == 'B' else w_val  
                g = img[h][w]
                interpolated_img[h][w] = [b,g,r]
            elif cfa_type[idx] == 'B':
                r = get_x(h,w)
                g = get_g(h,w)
                b = img[h][w]
                interpolated_img[h][w] = [b,g,r]

    interpolated_img = np.clip(interpolated_img, 0, 1)
    return interpolated_img


def contrast(img, magnitude):
    img = img / 255.0
    img = np.clip(img - magnitude * np.sin(2*(np.pi)*img) / (4 * np.pi), 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def unsharp(img, alpha):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb_img[:, :, 0].astype(np.float32)
    blurred_img = cv2.bilateralFilter(y, 9, 55, 55)
    ycrcb_img[:, :, 0] = np.clip(y + alpha * (y - blurred_img), 0, 255).astype(np.uint8)
    sharp_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return sharp_img

def saturation(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    filter = np.full(hsv_image.shape, (0, 20, 0), dtype=np.uint8)
    hsv_image = cv2.add(hsv_image, filter)
    hsv_image = np.clip(hsv_image, 0, 255)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Gamma correction
def gamma_correction(img, gamma):
    img = img / 255.0
    result_img = img ** (1 / gamma)
    result_img = (result_img * 255).astype(np.uint8)
    return result_img

def get_cfa_type(tiff_path):
    tags = exifread.process_file(open(tiff_path, 'rb'))
    metadata = {f"{tag}" : f"{value}" for tag, value in tags.items()}
    try:
        isVertical = metadata['Image ImageWidth'] < metadata['Image ImageLength']
        if metadata['Image Model'] == 'SM-G935L': # galaxy S7 edge
            return 'BGGR' if isVertical else 'GRBG'
        elif metadata['Image Model'] == 'SM-G991N': # galaxy S21
            return 'GRBG'
        else:
            raise Exception('Unknown Model')   
    except Exception as e:
        print('Unknown Model', e)

if __name__ == '__main__':
    tiff_path = sys.argv[1]
    filename = tiff_path.split('.')[0]
    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    img = img / (pow(2,16) - 1)
    cfa_type = get_cfa_type(tiff_path)
    balanced_img = white_balance(img, cfa_type)
    cv2.imwrite(f'../images/outputs/{filename}_balanced_img.jpg', (balanced_img * 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    interpolated_img = interpolation(balanced_img, cfa_type)
    interpolated_img = (interpolated_img * 255).astype(np.uint8)
    cv2.imwrite(f'../images/outputs/{filename}_interpolated_img.jpg', interpolated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    sharp_img = unsharp(interpolated_img, 1)
    cv2.imwrite(f'../images/outputs/{filename}_sharp_img.jpg', sharp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    color_enhanced_img = saturation(sharp_img)
    cv2.imwrite(f'../images/outputs/{filename}_color_enhanced_img.jpg', color_enhanced_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    result_img = gamma_correction(color_enhanced_img, 2.8)
    cv2.imwrite(f'../images/outputs/{filename}_gamma_correction.jpg', result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    result_img = contrast(result_img, 1.8)
    cv2.imwrite(f'../images/outputs/{filename}.jpg', result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    