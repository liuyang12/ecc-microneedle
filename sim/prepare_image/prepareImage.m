function [J, corners_raw] = prepareImage(I, params, warp, corners, I_bin)

if nargin < 2, params = PrepareImageParams(); end

J = I;

% Convert to grayscale.
if (size(J, 3) >= 3)
  J = rgb2gray(J(:, :, 1:3));
end

% Rescale to [0,1] range. [RESCALE]
J = double(J);
J = J / max(J(:));

% Rotate and crop the image to the center
% [naive] Crop around the brightest parts, assuming pattern is bright.
% [minrect] Minimum area rectangle of the binary thresholded image
[J, corners_raw] = imageCrop(J, params, warp, corners, I_bin);

% Resize to AlexNet compatible size.
J = imresize(J, params.dst_size); % original dst_size [227,227]

% Convert to 8-bit
J = uint8(255 * J);

% Make into 3 channels for AlexNet.
if params.num_channel > 1
    J = repmat(J, [1 1 params.num_channel]);
end

end
