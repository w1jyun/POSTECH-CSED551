blurred = imread('summerhouse.jpg');
blurred = im2double(blurred);

psf = imread('psf.png');
psf = im2double(psf);
psf = rgb2gray(psf);
psf = psf / sum(psf(:));

weight_x = ones(size(blurred));
weight_y = ones(size(blurred));

deblurred = deconv_L2(blurred, blurred, psf, 0.001, weight_x, weight_y);

imwrite(deblurred, 'deblurred.jpg');

