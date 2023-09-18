import cv2
import math
import numpy as np

from matplotlib import pyplot as plt

# for i in range(1, 7):
#     output = cv2.imread('images/output/histogram/gray%d_histogram.jpg' % i, cv2.IMREAD_COLOR)
#     input = cv2.imread('report/gray%d.jpg' % i, cv2.IMREAD_COLOR)
#     plt.subplot(1,3,1)
#     plt.imshow(input[:,:,::-1])
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.subplot(1,3,2)
#     plt.tight_layout()
#     plt.imshow(output[:,:,::-1]) # BGR -> RGB
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.tight_layout()
    
#     plt.tight_layout()
#     plt.savefig('report/' + 'gray_h%d' % i + '.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

for i in range(1, 7):
    output = cv2.imread('images/output/histogram/color%d_histogram.jpg' % i, cv2.IMREAD_COLOR)
    input = cv2.imread('report/color%d.jpg' % i, cv2.IMREAD_COLOR)
    plt.subplot(1,3,1)
    plt.imshow(input[:,:,::-1])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.tight_layout()
    plt.imshow(output[:,:,::-1]) # BGR -> RGB
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    
    plt.tight_layout()
    plt.savefig('report/' + 'color_h%d' % i + '.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


# for i in range(1, 7):
#     input = cv2.imread('images/input/color%d.jpg' % i, cv2.IMREAD_COLOR)
#     output = cv2.imread('images/output/histogram/color%d_result.jpg' % i, cv2.IMREAD_COLOR)
#     input = cv2.resize(input, dsize=(math.ceil(input.shape[1] * (512/input.shape[0])), 512), interpolation=cv2.INTER_AREA)
#     output = cv2.resize(output, dsize=(math.ceil(output.shape[1] * (512/output.shape[0])), 512), interpolation=cv2.INTER_AREA)
#     plt.subplot(1,2,1)
#     plt.imshow(input[:,:,::-1])
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.title('input', fontsize = 10)
#     plt.subplot(1,2,2)
#     plt.tight_layout()
#     plt.imshow(output[:,:,::-1]) # BGR -> RGB
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.title('output', fontsize = 10)
#     plt.tight_layout()
    
#     plt.tight_layout()
#     plt.savefig('report/' + 'color%d' % i + '.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


# for i in range(1, 7):
#     image = cv2.imread('report/color%d.jpg' % i, cv2.IMREAD_COLOR)
#     histogram = cv2.imread('images/output/histogram/color%d_histogram.jpg' % i, cv2.IMREAD_COLOR)

#     plt.subplot(2,1,1)
#     plt.imshow(histogram[:,:,::-1])
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.title('histogram', fontsize = 10)
#     plt.tight_layout()

#     plt.subplot(2,1,2)
#     plt.imshow(image[:,:,::-1]) # BGR -> RGB
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('report/' + 'h_color%d' % i + '.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


# for i in range(1, 4):
#     image = cv2.imread('report/gray_h%d.jpg' % i, cv2.IMREAD_COLOR)
#     plt.subplot(3,1,i)
#     plt.imshow(image[:,:,::-1])
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.tight_layout(pad=0.1)

# plt.savefig('report/' + 'grays1.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


# for i in range(4, 7):
#     image = cv2.imread('report/gray_h%d.jpg' % i, cv2.IMREAD_COLOR)
#     plt.subplot(3,1,i-3)
#     plt.imshow(image[:,:,::-1])
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.tight_layout(pad=0.1)

# plt.savefig('report/' + 'grays2.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
