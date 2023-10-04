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
    print(b*255, g*255, r*255)


    flt = ilf(filter_size, radius)
    plt.imsave('images/output/ilf/'+ image.split('.')[0] + '_filter_r_%d_%.1f_%d_result.jpg' % (filter_size, radius, border_type), flt, cmap = 'gray')
    flt_f = psf2otf(flt, (expanded_image.shape[0], expanded_image.shape[1]))
    spectrum_f = np.log(np.abs(flt_f))
    plt.imsave('images/output/ilf/'+ image.split('.')[0] + '_filter_freq_%d_%.1f_%d_result.jpg' % (filter_size, radius, border_type), np.fft.fftshift(spectrum_f), cmap = 'gray')

    results = []

    for i, c in enumerate([b, g, r]):
        img_f = np.fft.fft2(c)
        img_flt_f = flt_f * img_f 
        plt.imsave('images/output/ilf/' + image.split('.')[0] + '_img_spec_%d_%d_%.1f_%d_result.jpg' % (i, filter_size, radius, border_type), np.log(np.abs(np.fft.fftshift(img_f))), cmap = 'gray')
        # plt.imshow(np.fft.fftshift(np.log(np.abs(img_f))), cmap = 'gray')
        plt.close()

        img_flt = np.real(np.fft.ifft2(img_flt_f))
        plt.imsave('images/output/ilf/'+ image.split('.')[0] + '_img_spec_res_%d_%d_%.1f_%d_result.jpg' % (i, filter_size, radius, border_type), np.fft.fftshift(np.log(np.abs(img_flt_f))), cmap = 'gray')
        plt.close()
        img_flt = cv2.normalize(img_flt, None, np.min(c) * 255, np.max(c) * 255, cv2.NORM_MINMAX, -1)
        spectrum_f = np.log(np.abs(img_flt_f))
        results.append(img_flt)
    
    result_image = cv2.merge((results[0],results[1],results[2]))
    output = result_image[padding:-padding, padding:-padding, :]
    b, g, r = cv2.split(output)  
    for i, c in enumerate([b, g, r]):
            img_f = np.fft.fft2(c)
            plt.imsave('images/output/ilf/' + image.split('.')[0] + '_output_spec_%d_%d_%.1f_%d_result.jpg' % (i, filter_size, radius, border_type), np.fft.fftshift(np.log(np.abs(img_f))), cmap = 'gray')
            plt.close()


    cv2.imwrite('images/output/ilf/' + image.split('.')[0] + '_%d_%.1f_%d_result.jpg' % (filter_size, radius, border_type), output)
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

def getGaussianKernelValue(sigma, i, j):
    squared_sigma = pow(sigma, 2)
    return (1 / (2 * math.pi * squared_sigma)) * math.exp(-(pow(i,2) + pow(j,2)) / (2 * squared_sigma))

def getGaussianKernel2D(sigma, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    mid = math.floor(kernel_size/2)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = getGaussianKernelValue(sigma, -mid + i, -mid + j)
    alpha = 1 / sum(sum(kernel))
    return alpha * kernel

def gaussianFilteringFFT(image, filter_size, border_type):
    img = cv2.imread('images/input/'+image, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    padding = math.floor(filter_size / 2)
    expanded_image = cv2.copyMakeBorder(img, padding, padding, padding, padding, border_type)
    b, g, r = cv2.split(expanded_image)  

    flt = gauss2d((padding*2+1, padding*2+1), padding / 6.0)
    flt_f = psf2otf(flt, (expanded_image.shape[0], expanded_image.shape[1]))
    spectrum_f = np.log(np.abs(flt_f))
    plt.imsave('images/output/gaussian/'+ image.split('.')[0] + '_filter_freq_%d__%d_result.jpg' % (filter_size, border_type), np.fft.fftshift(spectrum_f), cmap = 'gray')
    plt.close()
    results = []
    for i, c in enumerate([b, g, r]):
        img_f = np.fft.fft2(c)
        img_flt_f = flt_f * img_f 
        plt.imsave('images/output/gaussian/' + image.split('.')[0] + '_img_spec_%d_%d__%d_result.jpg' % (i, filter_size, border_type), np.log(np.abs(np.fft.fftshift(img_f))), cmap = 'gray')
        plt.close()
        img_flt = np.real(np.fft.ifft2(img_flt_f))
        plt.imsave('images/output/gaussian/'+ image.split('.')[0] + '_img_spec_res_%d_%d__%d_result.jpg' % (i, filter_size, border_type), np.log(np.abs(np.fft.fftshift(img_flt_f))), cmap = 'gray')
        plt.close()
        img_flt = cv2.normalize(img_flt, None, np.min(c) * 255, np.max(c) * 255, cv2.NORM_MINMAX, -1)
        results.append(img_flt)

    result_image = cv2.merge((results[0],results[1],results[2]))
    output_image = result_image[padding:-padding, padding:-padding, :]
    b, g, r = cv2.split(output_image)  
    for i, c in enumerate([b, g, r]):
            img_f = np.fft.fft2(c)
            plt.imsave('images/output/gaussian/' + image.split('.')[0] + '_output_spec_%d_%d_%d_result.jpg' % (i, filter_size, border_type), np.fft.fftshift(np.log(np.abs(img_f))), cmap = 'gray')
            plt.close()
    cv2.imwrite('images/output/gaussian/' + image.split('.')[0] + '_%d__%d_result.jpg' % (filter_size, border_type), output_image)
    return output_image

def gaussianFilteringSpatial(image, filter_size, sigma, border_type):
    img = cv2.imread('images/input/'+image, cv2.IMREAD_COLOR)
    width = img.shape[0]
    height = img.shape[1]
    flt = getGaussianKernel2D(sigma, filter_size)

    padding = math.floor(filter_size/2)
    expanded_image = cv2.copyMakeBorder(img, padding, padding, padding, padding, border_type)
    output_image = np.zeros_like(img)
    for i in range(width):
        for j in range(height):
            sub_image = expanded_image[i:i+filter_size, j:j+filter_size].transpose()
            sub_image_b, sub_image_r, sub_image_g = sub_image[0].transpose(), sub_image[1].transpose(), sub_image[2].transpose()
            result_b = sum(sum(sub_image_b * flt))
            result_r = sum(sum(sub_image_r * flt))
            result_g = sum(sum(sub_image_g * flt))
            output_image[i][j] = np.array([result_b, result_r, result_g])
 
    return output_image

# promblem 3: UNSHARP MASKING & CONVOLUTION THEOREM
# two versions of unsharp masking:
    # Use spatial domain to implement unsharp masking
    # Use the frequency domain (use FFTs)
    # You need to implement all operations including convolution, addition, subtraction, and scaling in the frequency domain.
def unsharpMasking(image, alpha, sigma, domain):
    # 𝑌 = 𝑋 + 𝛼(𝑋 − 𝐺 ∗ 𝑋)
    # 𝑋: input image, 𝑌: sharpened result, 𝐺: low-pass filter(Gaussian), α: sharpening strength
    # size and shape of 𝐺 are controlled by sigma
    img = cv2.imread('images/input/'+image, cv2.IMREAD_COLOR)
    filter_size = sigma*6
    border_type = 1
    if domain == 'frequency':
        img = cv2.imread('images/input/'+image, cv2.IMREAD_COLOR).astype(np.float32)
        padding = math.floor(filter_size / 2)
        expanded_image = cv2.copyMakeBorder(img, padding, padding, padding, padding, border_type)
        b, g, r = cv2.split(expanded_image)  

        flt = getGaussianKernel2D(sigma, filter_size)
        flt_f = psf2otf(flt, (expanded_image.shape[0], expanded_image.shape[1]))
        spectrum_f = np.log(np.abs(flt_f))
        # plt.imshow(np.fft.fftshift(spectrum_f), cmap = 'gray')
        # plt.show()
        results = []

        for c in [b, g, r]:
            img_f = np.fft.fft2(c)
            spectrum = np.log(np.abs(img_f))
            img_flt_f = img_f + alpha * (img_f - flt_f * img_f)
            img_flt = np.real(np.fft.ifft2(img_flt_f))
            results.append(img_flt)

        result_image = cv2.merge((results[0],results[1],results[2]))
        output_image = result_image[padding:-padding, padding:-padding, :]
    else:
        gaussian_result = gaussianFilteringSpatial(image, filter_size, sigma, border_type)
        output_image = img + alpha * (img - gaussian_result)

    cv2.imwrite('images/output/unsharp/'+ domain + '/' + image.split('.')[0] + '_%d_%.1f_%d_result.jpg' % (filter_size, sigma, border_type), output_image)

for img in ['color3.jpg', 'shape2.jpg']:
    for s in [33, 77, 99]:
        gaussianFilteringFFT(img, s, 1)

# for img in ['color3.jpg', 'shape2.jpg']:
#     for r in [10, 20, 50]:
#         idealLowpassFiltering(img, 200, r, 1)

# gaussianFilteringFFT('color2.jpg', 99, 1, 1)
# gaussianFilteringFFT('color2.jpg', 99, 3, 1)
# unsharpMasking('color3.jpg', 10, 1, 'frequency')
# unsharpMasking('color3.jpg', 10, 1, 'gaussian')