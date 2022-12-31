function GenerateDotsRandomPatch( needle_side_size, num_images, dim_percent )
%GENERATEDOTSRANDOMPATCH Generate n x n randomly distributed microneedle
%array according to the template (four p x p corners for orientation)
%  input
%    needle_side_size   -   side size of the needle patch
%    num_images         -   total number of images to be generated
%    dim_percent        -   percentage of images with dimmer dots

if nargin <  1, needle_side_size = 10; end
if nargin <  2, num_images = 100;      end
if nargin <  3, dim_percent = 0.25;    end

% [0.0] directory configuration
addpath('../augment_image')
addpath('../prepare_image')

% [0.1] parameter configuration
prev_num_images = 0; % Previous number of images to generate per pattern

needle_params = NeedleParams(needle_side_size); 

needle_on_ratio_all = [1.0 , 0.75, 0.5 , 0.25];
ratio_percentage    = [0.05, 0.15, 0.6 , 0.2 ];

dim_dot_all             = [0, 1];
brightness_percent_all  = [1-dim_percent, dim_percent];
num_images_brightwise   = [0 cumsum(brightness_percent_all)] * num_images;

if num_images / 1e6 >= 1
    suffix = sprintf('%dM', num_images / 1e6);
elseif num_images / 1e3 >= 1
    suffix = sprintf('%dk', num_images / 1e3);
else
    suffix = sprintf('%d', num_images);
end

delete(gcp('nocreate')); % delete current parpool (if any)
mycluster = parcluster('local');
num_parfor_workers = round(mycluster.NumWorkers/2); 
poolobj = parpool(num_parfor_workers);

% configure outputs
images_rect = zeros([num_images needle_params.dst_size], "uint8");
labels = zeros([num_images needle_params.needle_array_size], "logical");

images_raw = zeros([num_images needle_params.rect_raw_size], "uint8");
corners = zeros([num_images 4 2], "double");

images_bin = zeros([num_images needle_params.rect_raw_size], "logical");

recog_output_file = sprintf('../output/needle%dx%d_%s_recog.mat',needle_params.needle_array_size(2),needle_params.needle_array_size(1),suffix);
rect_output_file = sprintf('../output/needle%dx%d_%s_rect.mat',needle_params.needle_array_size(2),needle_params.needle_array_size(1),suffix);
bin_output_file = sprintf('../output/needle%dx%d_%s_bin.mat',needle_params.needle_array_size(2),needle_params.needle_array_size(1),suffix);

if ~exist('../output/', 'dir')
    mkdir('../output/');
end

tic
fprintf('start generating %s random microneedle patches of size %dx%d ... \n', suffix, needle_side_size, needle_side_size);
for idim = 1:length(dim_dot_all)
    is_dim_dot = dim_dot_all(idim);
    brightness_percent = brightness_percent_all(idim);
    
    num_images_all      = round(ratio_percentage * num_images * brightness_percent);
    start_num_all       = round(ratio_percentage * prev_num_images * brightness_percent);

    num_images_stepwise = [0 cumsum(num_images_all)] + num_images_brightwise(idim);

    for iratio = 1:length(needle_on_ratio_all)
        needle_on_ratio = needle_on_ratio_all(iratio); % ratio of on pixels 
        num_images_current_ratio = num_images_stepwise(iratio);

        dot_params     = DotImageParams(is_dim_dot);
        augment_params = AugmentImageParams();
        prepare_params = PrepareImageParams(needle_params.dst_size);

        % [1] Template for 10x10 microneedle array (four 3x3 corners for orientation)
        [blank, mask] = MakeBlankNeedle(needle_params.needle_array_size, needle_params.corner_size, needle_params.corner_pos, needle_params.corner_hole, needle_params.allow_inner);
        idx = find(mask > 0);

        needle = blank; 

        start_num = start_num_all(iratio);
        
        images_rect_ratio = zeros([num_images_all(iratio) needle_params.dst_size], "uint8");
        labels_ratio = zeros([num_images_all(iratio) needle_params.needle_array_size], "logical");

        images_raw_ratio = zeros([num_images_all(iratio) needle_params.rect_raw_size], "uint8");
        corners_ratio = zeros([num_images_all(iratio) 4 2], "double");

        images_bin_ratio = zeros([num_images_all(iratio) needle_params.rect_raw_size], "logical");
        
        % [2] Generate random micrneedle array according to the template
        parfor (i = 1:num_images_all(iratio), num_parfor_workers)
        % for i = 1:num_images_all(iratio)
            k = i + num_images_current_ratio;

            randvec = rand(size(idx))>1-needle_on_ratio;
            needle = blank; 
            needle(idx) = randvec;

            % add random rotation 90 degrees to needle
            needle = rot90(needle, randi(4));

            [image_raw, num_dots, corner, image_bin] = DotImage(needle, dot_params);
            [image_raw, warp, image_bin] = AugmentImage(image_raw, image_bin, augment_params);
            [image, corner] = prepareImage(image_raw, prepare_params, warp, corner, image_bin);

            fprintf('  Image #%06d: %3d  / %3d dots; max %3d, min %3d.\n', k, num_dots, length(idx), max(image(:)), min(image(:)));

            images_rect_ratio(i,:,:) = image;
            labels_ratio(i,:,:) = needle;

            images_raw_ratio(i,:,:) = imresize(uint8(image_raw*255), needle_params.rect_raw_size);
            corners_ratio(i,:,:) = corner;

            images_bin_ratio(i,:,:) = imresize(image_bin, needle_params.rect_raw_size);
        end

        images_rect(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = images_rect_ratio;
        labels(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = labels_ratio;

        images_raw(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = images_raw_ratio;
        corners(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = corners_ratio;

        images_bin(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = images_bin_ratio;
    end
end

% save single .mat file in '-v7.3' version (>2GB)
save(recog_output_file, 'images_rect', 'labels', '-v7.3');
save(rect_output_file, 'images_raw', 'corners', '-v7.3');
save(bin_output_file, 'images_raw', 'images_bin', '-v7.3');

toc
delete(poolobj);
