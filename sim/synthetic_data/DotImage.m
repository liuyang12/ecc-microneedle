function [I, num_dots, corners, I_bin] = DotImage(shape, params)

dot_size_range = RandRange(params.dot_size_center_range) * params.dot_size_variation_range;

% Spatial support for each dot (6 standard deviations).
dot_spread = round(12 * max(dot_size_range) * max(params.glare_size_scale_range));

% Figure out the necessary image size.
shape_dim = size(shape, 1);
dot_max_offset = max(abs(params.dot_offset_range));
dot_distance = RandRange(params.dot_distance_range);
dot_distance_max = max(params.dot_distance_range);
image_dim = (shape_dim - 1) * dot_distance_max + dot_spread + 2 * dot_max_offset;
dot_center_offset = round((shape_dim - 1) * (dot_distance_max-dot_distance)/2);

I = zeros(image_dim);
I_bin = zeros(image_dim);

glare_size_scale = RandRange(params.glare_size_scale_range);
glare_brightness_scale = RandRange(params.glare_brightness_scale_range);

num_dots = 0;
pixel_range = 1:dot_spread;
for i = 1:shape_dim
  for j = 1:shape_dim
    if shape(i, j) > 0 && rand() >= params.p_missing
      num_dots = num_dots + 1;

      % Figure out the dot position.
      dot_offset = [RandRange(params.dot_offset_range), RandRange(params.dot_offset_range)];
      dot_position = dot_max_offset + dot_distance * [i-1, j-1] + dot_offset + dot_center_offset;
      dot_position = round(dot_position);

      % Generate the dot pixels with glare.
      dot_size = RandRange(dot_size_range);
      dot = fspecial('gaussian', dot_spread, dot_size);
      dot = dot / max(dot(:));
      glare_size = dot_size * glare_size_scale;
      glare = fspecial('gaussian', dot_spread, glare_size);
      glare = glare * glare_brightness_scale / max(glare(:));
      dot = dot + glare;
      dot_brightness = RandRange(params.dot_brightness_range);
      dot = dot * dot_brightness / max(dot(:));
      
      rows = pixel_range + dot_position(1);
      cols = pixel_range + dot_position(2);
      I(rows, cols) = I(rows, cols) + dot;
      
      % disk per dot as the ground-truth binarization
      dot_bin = get_disk_hsize(dot_size, dot_spread);
      dot_bin = dot_bin / max(dot_bin(:));
      dot_bin = dot_bin * dot_brightness / max(dot_bin(:));
      
      I_bin(rows, cols) = I_bin(rows, cols) + dot_bin;
    end
  end
end

% Scale to [0,1] range.
I = min(max(0.0, I / max(I(:))), 1.0);
I_bin = min(max(0.0, I_bin / max(I_bin(:))), 1.0);

% Adjust the overall brightness (exposure) of the image.
brightness_scale = RandRange(params.brightness_scale_range);
I     = brightness_scale * I;
I_bin = brightness_scale * I_bin;

% Four corners of the microneedleI patch (from top-left; clock-wise)
r = dot_spread/2;
% corners_idx = [1, 1; shape_dim, 1; shape_dim, shape_dim; 1, shape_dim]; % exact position
% corners_idx = [0, 0; shape_dim+1, 0; shape_dim+1, shape_dim+1; 0, shape_dim+1]; % expand one dot for each side ratio = (N+1)/(N-1)
corners_idx = [-.5, -.5; shape_dim+1.5, -.5; shape_dim+1.5, shape_dim+1.5; -.5, shape_dim+1.5]; % expand one dot for each side ratio = (N+2)/(N-1)
corners = dot_max_offset + dot_distance * (corners_idx-1) + [r, r] + dot_center_offset;

% absolute -> relative position of corners
corners = corners ./ size(I);

end


function spread = get_disk_hsize(dot_size, hsize)
    spread = zeros([hsize hsize]);
    dot = fspecial('disk', dot_size);
    c = size(dot,1);
    c1 = round((hsize-c)/2);
    spread(c1+(1:c),c1+(1:c)) = dot;
end