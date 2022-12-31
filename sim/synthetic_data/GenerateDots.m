clear; clc;
close all
addpath('../augment_image')
addpath('../prepare_image')

%  Number of images to generate per pattern.
% num_images = 15000;
num_images = 100;
% Should we save image files, or just display them?
save_files = true;
% target pixel resolution
dst_size = [120, 120];
% output_folder = '~/Projects/Tracking/Synthetic_Data';
% output_folder = 'D:\VT2\pmma_training';
% output_folder = '../output_resize';
output_folder = sprintf('../output/resize_%dx%d',dst_size(2),dst_size(1));

dot_params = DotImageParams();
augment_params = AugmentImageParams();
prepare_params = PrepareImageParams();
MakeShapes4x4

%ned of the for loop is "size(shapes, 3)"
for shape_index = 15:17
  shape = shapes(:, :, shape_index);
  folder = sprintf('%s/shape%d_%d_2', output_folder, shape_index, num_images);
  if (save_files && ~exist(folder,'dir'))
    mkdir(folder);
  end
  for i = 1:num_images
    disp(['shape ' num2str(shape_index) ' image ' num2str(i)]);
    [image, num_dots] = DotImage(shape, dot_params);
    image = AugmentImage(image, augment_params);
    % image = PrepareImage(image, num_dots); % [missing file] -> seem doing image resizing only
    image = prepareImage(image, prepare_params);
    
    fprintf('   max %d, min %d.\n', max(image(:)), min(image(:)));

    if (save_files)
      filename = sprintf('%s/shape%d_%05d_3.png', folder, shape_index, i);
      imwrite(image, filename);
    else
      imagesc(image);
      waitforbuttonpress;
    end
  end
end
