import cv2
import numpy as np
import matplotlib.pyplot as plt

def fftconv2(img, psf):
  return np.real(np.fft.ifft2(np.fft.fft2(img) * psf2otf(psf, (img.shape[0], img.shape[1]))))

def deconv_L2(blurred, latent0, psf, reg_strength):
    img_shape = blurred.shape
    latent = latent0.copy()
    # compute b
    b = fftconv2(blurred, np.rot90(psf, 2)) # (K^T)b
    b = b.flatten()

    # set x
    x = latent.flatten()

    # run conjugate gradient
    result = conjgrad(x, b, 25, 1e-4, Ax, psf, img_shape, reg_strength)
    latent = result.reshape(img_shape)
    return latent

def Ax(x, psf, img_shape, reg_strength):
    psf_conj = np.rot90(psf, 2)
    weight_x = np.ones(img_shape)
    weight_y = np.ones(img_shape)
    dxf = np.array([0, -1, 1])
    dyf = np.array([[0],[-1], [1]])

    x = x.reshape(img_shape)
    y = fftconv2(fftconv2(x, psf), psf_conj)
    y = y + reg_strength*cv2.filter2D(weight_x*x, -1, dxf)
    y = y + reg_strength*cv2.filter2D(weight_y*x, -1, dyf)
    # Compute Ax
    return y.flatten()

def conjgrad(x, b, maxIt, tol, Ax_func, psf, img_shape, reg_strength):
    r = b - Ax_func(x, psf, img_shape, reg_strength)
    p = r.copy()
    rsold = np.sum(r * r)

    for _ in range(1, maxIt + 1):
        Ap = Ax_func(p, psf, img_shape, reg_strength)
        alpha = rsold / np.sum(p * Ap)
        r = r - alpha * Ap
        rsnew = np.sum(r * r)
        x = x + alpha * p

        if np.sqrt(rsnew) < tol:
            break

        p = r + rsnew / rsold * p
        rsold = rsnew

    return x

# conjugate gradient method
def tv_deconv(file_name):
    blurred = cv2.imread('../images/examples/'+file_name+'.jpg', cv2.IMREAD_COLOR).astype(np.float32) / 255.
    kernel = cv2.imread('../images/examples/'+file_name+'_out.jpg.psf.png', 0).astype(np.float32) / 255.
    results = []
    psf = kernel / sum(sum(kernel))
    b, g, r = cv2.split(blurred)
    for x in [b, g, r]:
        results.append(255 * deconv_L2(x, x, psf, 0.001))
    result_image = cv2.merge((results[0],results[1],results[2]))
    cv2.imwrite('../images/outputs/tv/0.001/'+file_name+'.jpg', result_image)

def psf2otf(flt, img_shape):
    flt_top_half = flt.shape[0]//2
    flt_bottom_half = flt.shape[0] - flt_top_half
    flt_left_half = flt.shape[1]//2
    flt_right_half = flt.shape[1] - flt_left_half
    # Pad zeros to make the filter size the same as the image size
    flt_padded = np.zeros(img_shape, dtype=np.float32)
    # Shift the center to the top left corner
    flt_padded[:flt_bottom_half, :flt_right_half] = flt[flt_top_half:, flt_left_half:]
    flt_padded[:flt_bottom_half, img_shape[1]-flt_left_half:] = flt[flt_top_half:, :flt_left_half]
    flt_padded[img_shape[0]-flt_top_half:, :flt_right_half] = flt[:flt_top_half, flt_left_half:]
    flt_padded[img_shape[0]-flt_top_half:, img_shape[1]-flt_left_half:] = flt[:flt_top_half, :flt_left_half]
    # 2D FFT
    return np.fft.fft2(flt_padded)

def wiener_deconv(file_name, c_value):
    blurred = cv2.imread('../images/examples/'+file_name+'.jpg', cv2.IMREAD_COLOR).astype(np.float32) / 255.
    kernel = cv2.imread('../images/examples/'+file_name+'_out.jpg.psf.png', 0).astype(np.float32) / 255.
    b, g, r = cv2.split(blurred)
    psf = kernel / sum(sum(kernel))
    f_k = psf2otf(psf, (blurred.shape[0], blurred.shape[1]))
    results = []
    for i in [b,g,r]:
        f_i = np.fft.fft2(i)
        l_ = ((f_k ** 2) / (f_k ** 2 + c_value)) * (1 / f_k) * f_i
        results.append(np.real(np.fft.ifft2(l_)) * 255)

    result_image = cv2.merge((results[0],results[1],results[2]))

    cv2.imwrite('../images/outputs/wiener/'+str(c_value)+'/'+file_name+'.jpg', result_image)

for f in ['boy_statue', 'fishes', 'hanzi', 'harubang', 'summerhouse']:
    tv_deconv(f)
    for c in [10, 0.1, 0.01]:
        wiener_deconv(f, c)