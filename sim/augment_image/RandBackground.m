function background = RandBackground(image_size, params)
% generate glare and gradient background

% background_sigma_sum_range = [500,2000];
% background_amplitude_range = [0.2,0.8];
% background_warp_scale_range = [0.1,0.5];
% background_rotation_range = [-180,180];
% background_distortion_shift_range = [-200,200];
% background_sigma_range = [100,300];

background_sigma_sum_range = params.background_sigma_sum_range;
background_amplitude_range = params.background_amplitude_range;
background_warp_scale_range = params.background_warp_scale_range;
background_rotation_range = params.background_rotation_range;
background_distortion_shift_range = params.background_distortion_shift_range;
background_sigma_range = params.background_sigma_range;

H = image_size(1);
W = image_size(2);
% [X, Y] = meshgrid(1:W, 1:H);
% 
% x0 = round(RandRange([1,W]));
% y0 = round(RandRange([1,H]));

[X, Y] = meshgrid(1:2*W, 1:2*H);

x0 = round(RandRange([1,W]+W/2));
y0 = round(RandRange([1,H]+H/2));

sigma_sum = RandRange(background_sigma_sum_range);
xy_ratio = rand();
sigma_x = xy_ratio/(1+xy_ratio)*sigma_sum;
sigma_y = 1/(1+xy_ratio)*sigma_sum;
A = RandRange(background_amplitude_range);

background = gaussian2d(X, Y, A, x0, y0, sigma_x, sigma_y);

warp = RandomWarp(size(background), background_warp_scale_range, background_rotation_range, background_distortion_shift_range);
background = imwarp(background, warp, 'OutputView', imref2d(size(background)));

background = background((1:H)+floor(H/2),(1:W)+floor(W/2));

background = imgaussfilt(background, RandRange(background_sigma_range));
% filt_size = round(RandRange(background_sigma_range));
% background = medfilt2(background, [filt_size filt_size]);

end

% 2D Gaussian function
function f = gaussian2d(X, Y, A, x0, y0, sigma_x, sigma_y)
    f = A*exp(-((X-x0).^2/sigma_x^2 + (Y-y0).^2/sigma_y^2)/2);
end

