% dst_size     ---  destination size of the output image
% crop_method  ---  cropping method for the augmented simulation image 
% crop_margin  ---  cropping margin (extra size)
% num_channel  ---  number of output channels 
% homography_rect -- rectification based on ground truth homography (four corners with minor shift errors)
% corner_shift_range -- Range of corner shifts for homography-based rectification
function params = PrepareImageParams(...
    dst_size, ...
    fig_show, ...
    crop_method, ...
    crop_margin, ...
    filter_sigma, ...
    num_channel, ...
    corner_shift_range)

% Default values.
if nargin < 1, dst_size = [120, 120];   end  % original [227, 227]
if nargin < 2, fig_show = false;   end  % false
if nargin < 3, crop_method = 'homography'; end  % 'naive' | 'minrect' | homography
% if nargin < 3, crop_method = 'minrect'; end  % 'naive' | 'minrect' | homography
if nargin < 4, crop_margin = 0.35;      end  % original 0.25
if nargin < 5, filter_sigma = 11;       end  % original 11
if nargin < 6, num_channel = 1;         end  % original 3
% if nargin < 7, corner_shift_range = [-0, 0]; end % [optinal] 
if nargin < 7, corner_shift_range = [-10, 10]; end % [optinal] 

params.fig_show = fig_show;
params.dst_size = dst_size;
params.crop_method = crop_method;
params.crop_margin = crop_margin;
params.filter_sigma  = filter_sigma;
params.num_channel = num_channel;
params.corner_shift_range = corner_shift_range;

end
