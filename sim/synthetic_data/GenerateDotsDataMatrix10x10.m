clear; clc;
close all
addpath('../augment_image')
addpath('../prepare_image')

%  Number of images to generate per pattern.
num_images = 10;
is_dim_dot = 1;
% Should we save image files, or just display them?
save_files = true;
fig_show = true; % show figures of rectified image
homography_rect = true; % rectification based on ground truth homography (four corners with minor shift errors)

DataMatrix10x10

dst_size = ((size(shapes)+4)*2+2)*4; % target image pixel resolution

dot_params = DotImageParams(is_dim_dot);
augment_params = AugmentImageParams();
prepare_params = PrepareImageParams(dst_size,fig_show);

output_folder = sprintf('../output/datamatrix10x10/resize_%dx%d',dst_size(2),dst_size(1));

%ned of the for loop is "size(shapes, 3)"
for shape_index = 1:size(shapes,3)
  shape = 1-shapes(:, :, shape_index);
  folder = sprintf('%s/shape%d_%d_dim', output_folder, shape_index, num_images);
  if (save_files && ~exist(folder,'dir'))
    mkdir(folder);
  end
  for i = 1:num_images
    disp(['shape ' num2str(shape_index) ' image ' num2str(i)]);
    [image_raw, num_dots, corner, image_bin] = DotImage(shape, dot_params);
    [image_raw, warp, image_bin] = AugmentImage(image_raw, image_bin, augment_params);
    [image, corner] = prepareImage(image_raw, prepare_params, warp, corner, image_bin);
    
    fprintf('   max %d, min %d.\n', max(image(:)), min(image(:)));

    if (save_files)
      filename = sprintf('%s/shape%d_%05d.png', folder, shape_index, i);
      imwrite(image, filename);
    else
      imagesc(image); axis image;
      waitforbuttonpress;
    end
  end
end

%         % plot the corners of images before and after warping
%         figure(1); 
%         subplot(221); imshow(image); drawpolygon('Position',corners); title('original image'); 
%         subplot(222); imshow(augmented); drawpolygon('Position',corners_aug); drawpolygon('Position',corners_aug_shift,'Color','m'); title('warpped image');
% 
%         h = params.dst_size(1);
%         w = params.dst_size(2);
%         rect_pts = [1 1; w 1; w h; 1 h];rect_tform = fitgeotrans(corners_aug_shift, rect_pts, 'projective');
%         im_rect = imwarp(augmented, rect_tform, 'OutputView', imref2d(dst_size));
% 
%         subplot(223); imshow(imcrop(image, [corners(1,:),corners(3,:)-corners(1,:)])); title('cropped original image');
%         subplot(224); imshow(im_rect); title('warpped image after rectification');
