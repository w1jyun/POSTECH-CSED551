import cv2
import math
import numpy as np

from matplotlib import pyplot as plt
 
# problem 1: IDEAL LOWPASS FILTER

def psf2otf(flt, img_shape):
    flt_top_half = flt.shape[0]//2
    flt_bottom_half = flt.shape[0] - flt_top_half
    flt_left_half = flt.shape[1]//2
    flt_right_half = flt.shape[1] - flt_left_half
    # Pad zeros to make the filter size the same as the image size
    flt_padded = np.zeros(img_shape, dtype=flt.dtype)
    # Shift the center to the top left corner
    flt_padded[:flt_bottom_half, :flt_right_half] = flt[flt_top_half:, flt_left_half:]
    flt_padded[:flt_bottom_half, img_shape[1]-flt_left_half:] = flt[flt_top_half:, :flt_left_half]
    flt_padded[img_shape[0]-flt_top_half:, :flt_right_half] = flt[:flt_top_half, flt_left_half:]
    flt_padded[img_shape[0]-flt_top_half:, img_shape[1]-flt_left_half:] = flt[:flt_top_half, :flt_left_half]
    # 2D FFT
    return np.fft.fft2(flt_padded)

def ilf(filter_size, radius):
    # Distance from (u,v) to the center of the mask
    center = filter_size / 2
    filter = np.zeros((filter_size, filter_size))
    for u in range(filter_size):
        for v in range(filter_size):
            D = math.sqrt(pow((u-center), 2) + pow((v-center), 2))
            filter[u][v] = 0 if D > radius else 1.0
    return filter

def idealLowpassFiltering(image, filter_size, radius, border_type):
    img = cv2.imread('images/input/'+image, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    padding = math.floor(filter_size / 2)
    expanded_image = cv2.copyMakeBorder(img, padding, padding, padding, padding, border_type)
    b, g, r = cv2.split(expanded_image)  

    flt = ilf(filter_size, radius)
    flt_f = psf2otf(flt, (expanded_image.shape[0], expanded_image.shape[1]))

    results = []

    for c in [b, g, r]:
        img_f = np.fft.fft2(c)
        img_flt_f = flt_f * img_f 

        img_flt = np.real(np.fft.ifft2(img_flt_f))
        img_flt = cv2.normalize(img_flt, None, np.min(c) * 255, np.max(c) * 255, cv2.NORM_MINMAX, -1)
        results.append(img_flt)
    
    result_image = cv2.merge((results[0],results[1],results[2]))
    output = result_image[padding:-padding, padding:-padding, :]
    b, g, r = cv2.split(output)  

    cv2.imwrite('images/output/ideal/' + image.split('.')[0] + '_%d_%.1f_%d_result.jpg' % (filter_size, radius, border_type), output)
    return output

# problem 2: GAUSSIAN LOWPASS FILTER

def gauss(n,sigma):
    r = np.arange(0, n, dtype=np.float32) - (n-1.)/2.
    r = np.exp(-r**2./(2.*sigma**2))
    return r / np.sum(r)

def gauss2d(shape, sigma):
    g1 = gauss(shape[0], sigma).reshape([shape[0], 1])
    g2 = gauss(shape[1], sigma).reshape([1, shape[1]])
    return np.matmul(g1,g2)

def gaussianFilteringFFT(image, filter_size, border_type):
    img = cv2.imread('images/input/'+image, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    padding = math.floor(filter_size / 2)
    expanded_image = cv2.copyMakeBorder(img, padding, padding, padding, padding, border_type)
    b, g, r = cv2.split(expanded_image)  

    flt = gauss2d((padding*2+1, padding*2+1), padding / 6.0)
    flt_f = psf2otf(flt, (expanded_image.shape[0], expanded_image.shape[1]))

    results = []
    for i, c in enumerate([b, g, r]):
        img_f = np.fft.fft2(c)
        img_flt_f = flt_f * img_f 
        img_flt = np.real(np.fft.ifft2(img_flt_f))
        img_flt = cv2.normalize(img_flt, None, np.min(c) * 255, np.max(c) * 255, cv2.NORM_MINMAX, -1)
        results.append(img_flt)

    result_image = cv2.merge((results[0],results[1],results[2]))
    output_image = result_image[padding:-padding, padding:-padding, :]
    b, g, r = cv2.split(output_image)  
    cv2.imwrite('images/output/gaussian/' + image.split('.')[0] + '_%d__%d_result.jpg' % (filter_size, border_type), output_image)
    return output_image

# promblem 3: UNSHARP MASKING & CONVOLUTION THEOREM
# two versions of unsharp masking:
    # Use spatial domain to implement unsharp masking
    # Use the frequency domain (use FFTs)
def unsharpMasking(image, alpha, filter_size, domain, border_type):
    # Y = X + a(X-G*X)
    # X: input image, Y: sharpened result, G: low-pass filter(Gaussian), a: sharpening strength
    # size and shape of G are controlled by sigma
    img = cv2.imread('images/input/'+image, cv2.IMREAD_COLOR)
    padding = math.floor(filter_size / 2)
    expanded_image = cv2.copyMakeBorder(img, padding, padding, padding, padding, border_type)
    flt = gauss2d((padding*2+1, padding*2+1), padding / 6.0)
    results = []
    output_image = np.zeros_like(img)
    if domain == 'frequency':
        flt_f = psf2otf(flt, (expanded_image.shape[0], expanded_image.shape[1]))
        b, g, r = cv2.split(expanded_image)  
        for c in [b, g, r]:
            img_f = np.fft.fft2(c)
            img_flt_f = img_f + alpha * (img_f - flt_f * img_f)
            img_flt = np.real(np.fft.ifft2(img_flt_f))
            results.append(img_flt)

        result_image = cv2.merge((results[0],results[1],results[2]))
        output_image = result_image[padding:-padding, padding:-padding, :]
        cv2.imwrite('images/output/unsharp/'+ domain + '/' + image.split('.')[0] + '_%d_%d_%d_result.jpg' % (filter_size, alpha, border_type), output_image)
    else:
        width = img.shape[0]
        height = img.shape[1]
        img_flt = np.zeros_like(img).astype(np.float32) 
        for i in range(width):
            for j in range(height):
                sub_image = expanded_image[i:i+filter_size, j:j+filter_size].astype(np.float32) 
                b, g, r = cv2.split(sub_image) 
                for i_c, c in enumerate([b, g, r]):
                    img_flt[i][j][i_c] = sum(sum(flt * c))

        output_image = img + alpha * (img - img_flt)
        cv2.imwrite('images/output/unsharp/'+ domain + '/' + image.split('.')[0] + '_%d_%d_%d_result.jpg' % (filter_size, alpha, border_type), output_image)
    
    return output_image

for img in ['color8.jpg','shape.jpg']:
    # for size in [10, 30, 50]:
    #     idealLowpassFiltering(img, size * 3, size, 1)

    # for filer in [33,77,99]:
    #     gaussianFilteringFFT(img, filer, 1)

    for domain in ['spatial']:
        for alpha in [1.0, 2.0, 3.0]:
            for filter_size in [33,77,99]:
                print(alpha, filter_size)
                unsharpMasking(img, alpha, filter_size, domain, 1)