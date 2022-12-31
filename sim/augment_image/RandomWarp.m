function [T, T_inv] = RandomWarp(image_size, warp_scale_range, rot_degree_range, distortion_shift_range)

if nargin < 3, rot_degree_range = [-180, 180]; end
if nargin < 4, distortion_shift_range = [-0, 0]; end

h = image_size(1); h2 = round(h/2);
w = image_size(2); w2 = round(w/2);
% from = [1 1 1; w 1 1; 1 h 1; w h 1]'; % [four corners] - projective
from = [1 1 1; w 1 1; 1 h 1; w h 1;
        w2 1 1; w h2 1; w2 h 1; 1 h2 1]'; % [four corners + four centers of sides] - polynomial

% apply random rotation
R = RandomRotation(image_size, rot_degree_range);
to = R * from;

% apply random offsets
warp_scale = RandRange(warp_scale_range);
% fprintf('    warp scale %.3f.', warp_scale);
to = to + warp_scale * norm(image_size) * RandCentered(size(from));

% [TODO] add geometric transform 2D (sinusoidal to model curved surface)
% reference: https://www.mathworks.com/help/images/creating-a-gallery-of-transformed-images.html

if max(distortion_shift_range) > 0
    % to(1:2, 5:end) = to(1:2, 5:end) + rand([2,4])*max(distortion_shift_range);
    T = fitgeotrans(to(1:2, :)', from(1:2, :)', 'polynomial', 2);
    T_inv = fitgeotrans(from(1:2, :)', to(1:2, :)', 'polynomial', 2);
else
    T = fitgeotrans(to(1:2, :)', from(1:2, :)', 'projective');
    T_inv = fitgeotrans(from(1:2, :)', to(1:2, :)', 'projective');
end

end