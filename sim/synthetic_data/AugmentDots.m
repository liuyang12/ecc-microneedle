clear;
clc;
addpath('../augment_image')
addpath('../prepare_image')

% Where to find source images.
image_folder = '~/Projects/Tracking/Data2/';
% Number of augmented images for each source image.
num_augmented_images = 1;
% Should we save image files, or just display them?
save_files = true;
output_folder = '~/Projects/Tracking/Augmented_Data';
if (save_files)
  mkdir(output_folder);
end

% Parameters that specify how to augment images.
augment_params = AugmentImageParams();

ds = datastore(image_folder, 'Type', 'image');
image_filenames = ds.Files;
num_source_images = length(image_filenames);

for i = 1:num_source_images
  filename = image_filenames{i};
  I = imread(filename);
  if max(I(:)) - 5 * (median(I(:)) + 1) < 32
    disp(['Image ' image_filenames{i} ' does not have enough contrast']);
    continue;
  end
  I = rgb2gray(I);
  I = double(I) / 255;
  [file_path, file_name, file_ext] = fileparts(filename); 
  for j = 1:num_augmented_images
    disp([filename ' ' num2str(j)]);
    J = AugmentImage(I, augment_params);
    J = PrepareImage(J);
    if (save_files)
      out_filename = sprintf('%s/%s_%02d%s', output_folder, file_name, j, file_ext);
      imwrite(J, out_filename);
    else
      imshow(J);
      waitforbuttonpress;
    end
  end
end
