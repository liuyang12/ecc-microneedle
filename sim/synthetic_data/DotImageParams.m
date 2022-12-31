% is_dim_dot -- generating dimmer dots with desirable results
% dot_distance_range -- Range of nominal distance between dots in the grid.
% [deprecated] dot_size_range -- Standard deviation of dot gaussians is uniform random in this range.
% dot_size_center_range -- center of the range of dot gaussians' standard deviation
% dot_size_variation_range -- variation in percentage of the range of dot gaussians' standard deviation (referring to the center)
% dot_brightness_range -- Max brightness of dot gaussians is uniform random in this range.
% dot_offset_range -- Horizontal and vertical dot offsets are uniform randoms in this range.
% glare_size_scale_range -- Standard deviation of glare gaussians is uniform random in this range, multiplied by dot size.
% glare_brightness_scale_range -- Max brightness of glare gaussians is uniform random in this range, multiplied by dot brightness.
% p_missing -- Probability of a dot missing.
% resolution_scale -- All sizes are scaled by this number (to produce higher resolution data).
% brightness_scale_range -- Min and max value to scale image brightness by.
function params = DotImageParams( ...
  is_dim_dot,                     ...
  dot_distance_range,             ...
  dot_size_center_range,          ...
  dot_size_variation_range,       ...
  dot_offset_range,               ...
  dot_brightness_range,           ...
  glare_size_scale_range,         ...
  glare_brightness_scale_range,   ...
  p_missing,                      ...
  resolution_scale,               ...
  brightness_scale_range)

% Default values.
% [TODO: division of large and small dots]
if nargin < 1, is_dim_dot = 0; end
if nargin < 2, dot_distance_range = [4, 10]; end
if nargin < 3, dot_size_center_range = [0.5, 2.5]; end % original value [.5, 1.5]
if nargin < 4, dot_size_variation_range = [0.5, 1.35]; end % original value [.5, 1.5]
if nargin < 5, dot_offset_range = [-1, 1]; end % original value [-.5,.5]
if nargin < 7, glare_size_scale_range = [1.0 10.0]; end
if nargin < 9, p_missing = 0; end % original value=0.1
if nargin < 10, resolution_scale = 4; end

if is_dim_dot
    % dimmer dots
    if nargin < 6, dot_brightness_range = [0.05, 0.2]; end % original value regular [0.5, 1.0] / dimmer [0.1, 0.3]
    if nargin < 8, glare_brightness_scale_range = [0.02, 0.1]; end
    if nargin < 11, brightness_scale_range = [0.25, 0.5]; end
else
    % ordinary dots
    if nargin < 6, dot_brightness_range = [0.5, 1.1]; end % original value [0.5, 1.0]
    if nargin < 8, glare_brightness_scale_range = [0.1, 0.3]; end
    if nargin < 11, brightness_scale_range = [0.5, 1.25]; end
end

params.is_dim_dot = is_dim_dot;
params.dot_distance_range = dot_distance_range;
% params.dot_size_range = dot_size_range;
params.dot_size_center_range = dot_size_center_range;
params.dot_size_variation_range = dot_size_variation_range;
params.dot_offset_range = dot_offset_range;
params.dot_brightness_range = dot_brightness_range;
params.glare_size_scale_range = glare_size_scale_range;
params.glare_brightness_scale_range = glare_brightness_scale_range;
params.p_missing = p_missing;
params.brightness_scale_range = brightness_scale_range;

% Scale spatial distances by resolution scale.
params.dot_distance_range = resolution_scale * params.dot_distance_range;
% params.dot_size_range = resolution_scale * params.dot_size_range;
params.dot_size_center_range = resolution_scale * params.dot_size_center_range;
params.dot_offset_range = ceil(resolution_scale * params.dot_offset_range);

end
