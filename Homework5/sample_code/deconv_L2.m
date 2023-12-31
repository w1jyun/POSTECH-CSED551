%  Non-blind deconvolution using a regulrization term based on an L2 norm on the image gradient
%
% PARAMETERS
%  - blurred: blurred image
%  - latent0: initial solution for latent
%  - psf: blur kernel
%  - reg_strength: regularization strength
%  - weight_x, weight_y: weight maps for the IRLS method
function latent = deconv_L2(blurred, latent0, psf, reg_strength, weight_x, weight_y)
    img_size = size(blurred);

    dxf=[0 -1 1];
    dyf=[0 -1 1]';
    
    latent = latent0;

    psf_conj = rot90(psf, 2);

    % compute b
    b = fftconv2(blurred, psf_conj);
    b = b(:);

    % set x
    x = latent(:);

    % run conjugate gradient
    cg_param.psf = psf;
    cg_param.psf_conj = psf_conj;
    cg_param.img_size = img_size;
    cg_param.reg_strength = reg_strength;
    cg_param.weight_x = weight_x;
    cg_param.weight_y = weight_y;
    cg_param.dxf = dxf;
    cg_param.dyf = dyf;
    x = conjgrad(x, b, 25, 1e-4, @Ax, cg_param); %, @vis);
    
    latent = reshape(x, img_size);
end

function ret = fftconv2(img, psf)
  ret = real(ifft2(fft2(img) .* psf2otf(psf, size(img))));
end

function y = Ax(x, p)
    x = reshape(x, p.img_size);
    x_f = fft2(x);
    y = fftconv2(fftconv2(x, p.psf), p.psf_conj);
    y = y + p.reg_strength*imfilter(p.weight_x.*imfilter(x, p.dxf, 'circular'), p.dxf, 'conv', 'circular');
    y = y + p.reg_strength*imfilter(p.weight_y.*imfilter(x, p.dyf, 'circular'), p.dyf, 'conv', 'circular');
    y = y(:);
end

function vis(x, iter, p)
    if mod(iter,5) == 0
        x = reshape(x, p.img_size);
        x = x(1:end-(size(p.psf,1)-1), 1:end-(size(p.psf,2)-1), :);
        figure(1), imshow(x), title(sprintf('%d',iter));
        drawnow;
    end
end
