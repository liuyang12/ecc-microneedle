function [augmented, warp, image_bin] = AugmentImage(image, image_bin, params)

augmented = double(image);

% Warp the image.
warp = RandomWarp(size(augmented), params.warp_scale_range, params.rot_degree_range, params.distortion_shift_range);
augmented = imwarp(augmented, warp, 'OutputView', imref2d(size(augmented)));
% figure(2); subplot(222); imshow(augmented>max(augmented(:))*0.3); title('binarized after random warping (noise-free)');
image_bin = imwarp(image_bin, warp, 'OutputView', imref2d(size(image_bin)));
image_bin = image_bin > 0;
% figure(2); subplot(221); imshow(augmented); title('after random warping (noise-free)');
% figure(2); subplot(222); imshow(image_bin>0); title('binarized after random warping (noise-free)');

% Add global glare to the whole image
background_highlight = RandBackground(size(augmented), params);
params.background_sigma_sum_range = [400,600];
params.background_amplitude_range = [0.2,0.4];
background_glare = RandBackground(size(augmented), params);
augmented = augmented + background_highlight + background_glare;
% figure(2); subplot(223); imshow(augmented); title('after random warping (w/ background glare)');

% Add defocus and motion blur
defocus_sigma = RandRange(params.defocus_sigma_range);
augmented = imgaussfilt(augmented, defocus_sigma);

motion_length = RandRange(params.motion_length_range);
motion_degree = RandRange(params.motion_degree_range);
motion_kernel = fspecial('motion', motion_length, motion_degree);
augmented = imfilter(augmented, motion_kernel, 'replicate');
% figure(2); subplot(224); imshow(augmented); title('after random warping (w/ defocus/motion blur)');

% Add random noise.
noise = RandomNoise(size(augmented), params.noise_weights);
noise_mean = RandRange(params.noise_mean_range);
noise_deviation = RandRange(params.noise_deviation_range);
%noise =  noise_mean + noise_deviation * sqrt(pi / 2) * abs(noise);
noise =  noise_mean + noise_deviation * noise;
augmented = augmented + noise;

% Clamp to [0,1] range.
augmented = min(max(0.0, augmented), 1.0);

end


