import cv2
import math
import numpy as np

from matplotlib import pyplot as plt
 
# problem 1: GAUSSIAN FILTERING
# [input]
# - image: input RGB image
# - kernel_size: an odd integer to specify the kernel size. If this is 5, then the actual kernel size is 5×5.
# - kernel_sigma: a positive real value to control the shape of the filter kernel.
# - border_type: extrapolation method for handling image boundaries.
#   - Possible values are: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101
# separable: Boolean value. If separable == true, then the function performs Gaussian filtering using two 1D filters. Otherwise, the function performs Gaussian filtering using a normal 2D convolution operation.
# [output]
# - filtered RGB image

def getGaussianKernelValue(sigma, i, j):
    squared_sigma = pow(sigma, 2)
    return (1 / (2 * math.pi * squared_sigma)) * math.exp(-(pow(i,2) + pow(j,2)) / (2 * squared_sigma))

def getGaussianKernel1D(sigma, kernel_size):
    row = np.zeros((1, kernel_size))
    for i in range(kernel_size):
        row[0][i] = pow(getGaussianKernelValue(sigma, -math.floor(kernel_size/2) + i, -math.floor(kernel_size/2) + i), 1/2)
    column = row.transpose()
    return row, column

def getGaussianKernel2D(sigma, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    mid = math.floor(kernel_size/2)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = getGaussianKernelValue(sigma, -mid + i, -mid + j)
    return kernel

def filterGaussian(image, kernel_size, kernel_sigma, border_type, seperable):
    img = cv2.imread('images/input/'+image, cv2.IMREAD_COLOR)
    width = img.shape[0]
    height = img.shape[1]
    if seperable:
        row, column = getGaussianKernel1D(kernel_sigma, kernel_size)
    else:
        kernel = getGaussianKernel2D(kernel_sigma, kernel_size)
    s = math.floor(kernel_size/2)
    expanded_image = cv2.copyMakeBorder(img, s, s, s, s, border_type)
    output_image = np.zeros_like(img)
    for i in range(width):
        for j in range(height):
            sub_image = expanded_image[i:i+kernel_size, j:j+kernel_size].transpose()
            result = (sub_image * column * row) if seperable else (sub_image * kernel)
            output_image[i][j] = np.array([np.sum(result[0]), np.sum(result[1]), np.sum(result[2])])

    cv2.imwrite('images/output/gaussian/' + image.split('.')[0] + '_%d_%.1f_%d_expand.jpg' % (kernel_size, kernel_sigma, border_type), expanded_image)
    cv2.imwrite('images/output/gaussian/' + image.split('.')[0] + '_%d_%.1f_%d_result.jpg' % (kernel_size, kernel_sigma, border_type), output_image)
    return output_image

for i in range(2,3):
    for size in [3, 33, 99]:
       for sigma in [1, 2, 3]:
            # t = 1
            for t in range(5):
                filterGaussian('shape%d.jpg'%i, size, sigma, t, True)

# problem 2: HISTOGRAM EQUALIZATION
# implement two versions of histogram equalization: grayscale-version, color-version.
# For the color version, you can simply apply the grayscale-version to each color channel independently.

def equalizeHistogramGrayscale(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    width = img.shape[0]
    height = img.shape[1]
    histogram = np.zeros(256)
    for i in range(width):
        for j in range(height):
            v = img[i][j]
            histogram[v] += 1
    
    cumulative_histogram = np.zeros(256)
    cumulative_histogram[0] = histogram[0]
    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i-1] + histogram[i]
    alpha = (width * height) / 255
    def transform(x):
        return (1 / alpha) * cumulative_histogram[x]

    output_image = np.zeros_like(img)
    for i in range(width):
        for j in range(height):
            output_image[i][j] = transform(img[i][j])

    # plt.subplot(3,1,1)
    # plt.hist(img.reshape(-1,1), bins=255, cumulative=False)
    # plt.title('histogram(before)')
    # plt.subplot(3,1,2)
    # plt.hist(img.reshape(-1,1), bins=255, cumulative=True)
    # plt.title('cumulative histogram')
    # plt.subplot(3,1,3)
    # plt.hist(output_image.reshape(-1,1), bins=255)
    # plt.title('histogram(after)')
    # plt.tight_layout()
    # plt.close('all')
    return output_image

def equalizeHistogramColorscale(image):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    width = img.shape[0]
    height = img.shape[1]
    histogram = np.zeros((256, 3))
    for i in range(width):
        for j in range(height):
            v = img[i][j]
            histogram[v[0]][0] += 1
            histogram[v[1]][1] += 1
            histogram[v[2]][2] += 1
    
    cumulative_histogram = np.zeros((256, 3))
    cumulative_histogram[0] = histogram[0]
    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i-1] + histogram[i]
    alpha = (width * height) / 255

    def transform(x, i):
        return np.multiply(cumulative_histogram[x[i]][i], (1 / alpha))

    output_image = np.zeros_like(img)
    for i in range(width):
        for j in range(height):
            output_image[i][j] = np.array([transform(img[i][j], 0), transform(img[i][j], 1), transform(img[i][j], 2)])

    # plt.subplot(3,1,1)
    # plt.hist(img.transpose()[0].reshape(-1,1), histtype='step', bins=256, color='blue')
    # plt.hist(img.transpose()[1].reshape(-1,1), histtype='step', bins=256, color='green')
    # plt.hist(img.transpose()[2].reshape(-1,1), histtype='step', bins=256, color='red')
    # plt.title('histogram(before)')
    # plt.subplot(3,1,2)
    # plt.hist(img.transpose()[0].reshape(-1,1), histtype='step', cumulative=True, bins=256, color='blue')
    # plt.hist(img.transpose()[1].reshape(-1,1), histtype='step', cumulative=True, bins=256, color='green')
    # plt.hist(img.transpose()[2].reshape(-1,1), histtype='step', cumulative=True, bins=256, color='red')
    # plt.title('cumulative histogram')
    # plt.subplot(3,1,3)
    # plt.hist(output_image.transpose()[0].reshape(-1,1), histtype='step', bins=256, color='blue')
    # plt.hist(output_image.transpose()[1].reshape(-1,1), histtype='step', bins=256, color='green')
    # plt.hist(output_image.transpose()[2].reshape(-1,1), histtype='step', bins=256, color='red')
    # plt.title('histogram(after)')
    # plt.tight_layout()
    # plt.savefig('images/output/histogram/color%d_histogram.jpg'%index, dpi=300)
    # plt.close('all')

    return output_image
