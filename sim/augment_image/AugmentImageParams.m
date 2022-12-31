% warp_scale_range -- Range of max proportion of image size to move the four corners of the image.
% distortion_shift_range -- Range of distortion shift (four centers of )
% rot_degree_range -- Range of rotation degrees 
% defocus_sigma_range -- Range of defocus blur std (sigma)
% motion_length_range -- Range of motion blur kernel length
% motion_degree_range -- Range of motion blur orientation (w.r.t. horizontal lines)
% noise_mean_range -- Min and max value of the noise mean.
% noise_deviation_range -- Min and max value of the noise standard deviation.
% noise_weights -- How much to scale each frequency level of noise.
% 
% background_sigma_sum_range - 
% background_amplitude_range - 
% background_warp_scale_range - 
% background_rotation_range - 
% background_distortion_shift_range - 

% aspect_ratio_range - range of the target aspect ratio of the raw image
function params = AugmentImageParams(...
  warp_scale_range, ...
  distortion_shift_range, ...
  rot_degree_range, ...
  defocus_sigma_range, ...
  motion_length_range, ...
  motion_degree_range, ...
  noise_mean_range, ...
  noise_deviation_range, ...
  noise_weights, ...
  background_sigma_sum_range, ...
  background_amplitude_range, ...
  background_warp_scale_range, ...
  background_rotation_range, ...
  background_distortion_shift_range, ...
  background_sigma_range, ...
  aspect_ratio_range)

% Default values.
if nargin <  1, warp_scale_range = [0.04, 0.2]; end
if nargin <  2, distortion_shift_range = [-0, 0]; end
if nargin <  3, rot_degree_range = [-10, 10]; end % [optinal] original [-180, 180] -> [-5, 5]
if nargin <  4, defocus_sigma_range = [1, 4]; end % [optional]
if nargin <  5, motion_length_range = [1, 8]; end % [optional]
if nargin <  6, motion_degree_range = [0, 360]; end % [optional]
if nargin <  7, noise_mean_range = [0.02, 0.12]; end
if nargin <  8, noise_deviation_range = [0.01, 0.03]; end
if nargin <  9, noise_weights = [1 1 1 1]; end
if nargin < 10, background_sigma_sum_range = [500,1500]; end
if nargin < 11, background_amplitude_range = [0.3,0.6]; end
if nargin < 12, background_warp_scale_range = [0.05,0.2]; end
if nargin < 13, background_rotation_range = [-180,180]; end
if nargin < 14, background_distortion_shift_range = [-25,25]; end
if nargin < 15, background_sigma_range = [50,100]; end
if nargin < 16; aspect_ratio_range = [1.5,1.5]; end


params.warp_scale_range = warp_scale_range;
params.distortion_shift_range = distortion_shift_range;
params.rot_degree_range = rot_degree_range;
params.defocus_sigma_range = defocus_sigma_range;
params.motion_length_range = motion_length_range;
params.motion_degree_range = motion_degree_range;
params.noise_mean_range = noise_mean_range;
params.noise_deviation_range = noise_deviation_range;
params.noise_weights = noise_weights;

params.background_sigma_sum_range = background_sigma_sum_range;
params.background_amplitude_range = background_amplitude_range;
params.background_warp_scale_range = background_warp_scale_range;
params.background_rotation_range = background_rotation_range;
params.background_distortion_shift_range = background_distortion_shift_range;
params.background_sigma_range = background_sigma_range;

params.aspect_ratio_range = aspect_ratio_range;

end
