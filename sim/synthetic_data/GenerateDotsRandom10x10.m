%GENERATEDOTSRANDOM10x10 Generate 10x10 randomly distributed microneedle
%array according to the template (four 3x3 corners for orientation)

clear; clc;
close all

% [0.0] directory configuration
addpath('../augment_image')
addpath('../prepare_image')

% [0.1] parameter configuration
num_images = 1e6; % Number of images to generate per pattern
prev_num_images = 0; % Previous number of images to generate per pattern
dim_percent = 0.25; % Percent of images with dimmer dots
save_single_file = true; % save a single file or individual image files as the dataset
save_figs = false;  % save figures
homography_rect = true; % rectification based on ground truth homography (four corners with minor shift errors)

needle_array_size = [10, 10]; % microneedle array sizeblank = zeros(needle_array_size, 'logical');
rect_raw_size = [256, 256]; % raw image size for rectification
corner_size = 3; % p - corner size pxp for orientation
corner_pos = 'southeast'; % position of the pxp orientation pixels 
corner_hole = true; % has hole in the corner pixels or not
allow_inner = false; % allow innner pixels

needle_on_ratio_all = [1.0 , 0.75, 0.5 , 0.25];
ratio_percentage    = [0.05, 0.15, 0.6 , 0.2 ];

dst_size = ((needle_array_size+4)*2+2)*4; % target image pixel resolution

mycluster = parcluster('local');
delete(gcp('nocreate')); % delete current parpool (if any)
num_parfor_workers = round(mycluster.NumWorkers/2); 
poolobj = parpool(num_parfor_workers);

% suffix = '_all_40k'; % [2%; 18%; 60%; 20%] * 500k
% suffix = '_all_100k'; % [2%; 18%; 60%; 20%] * 100k
% suffix = '_all_500k'; % [2%; 18%; 60%; 20%] * 500k
% suffix = '_dim_50k'; % [2%; 18%; 60%; 20%] * 50k
% suffix = '_ord_500k'; % new dataset [100 - 10%; 75 - 20%; 50 - 40%; 25 - 20%]

dim_dot_all             = [0, 1];
brightness_percent_all  = [1-dim_percent, dim_percent];

if num_images / 1e6 >= 1
    suffix = sprintf('_%dM', num_images / 1e6);
elseif num_images / 1e3 >= 1
    suffix = sprintf('_%dk', num_images / 1e3);
else
    suffix = sprintf('_%d', num_images);
end

% configure outputs
if save_single_file
    images_rect = zeros([num_images dst_size], "uint8");
    labels = zeros([num_images needle_array_size], "logical");

    images_raw = zeros([num_images rect_raw_size], "uint8");
    corners = zeros([num_images 4 2], "double");

    images_bin = zeros([num_images rect_raw_size], "logical");

    recog_output_file = sprintf('../output/needle%dx%d%s_recog.mat',needle_array_size(2),needle_array_size(1),suffix);
    rect_output_file = sprintf('../output/needle%dx%d%s_rect.mat',needle_array_size(2),needle_array_size(1),suffix);
    bin_output_file = sprintf('../output/needle%dx%d%s_bin.mat',needle_array_size(2),needle_array_size(1),suffix);

    if ~exist('../output/', 'dir')
        mkdir('../output/');
    end
else
    output_image_folder = sprintf('../output/needle%dx%d_image%dx%d%s/image',needle_array_size(2),needle_array_size(1),dst_size(2),dst_size(1),suffix);
    if ~exist(output_image_folder, 'dir')
        mkdir(output_image_folder);
    end
    output_mask_folder = sprintf('../output/needle%dx%d_image%dx%d%s/mask',needle_array_size(2),needle_array_size(1),dst_size(2),dst_size(1),suffix);
    if ~exist(output_mask_folder, 'dir')
        mkdir(output_mask_folder);
    end

    if save_figs
        output_fig_folder = sprintf('../output/needle%dx%d_image%dx%d%s/fig',needle_array_size(2),needle_array_size(1),dst_size(2),dst_size(1),suffix);
        if ~exist(output_fig_folder, 'dir')
            mkdir(output_fig_folder);
        end
    end
end

num_images_brightwise = [0 cumsum(brightness_percent_all)] * num_images;
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
        prepare_params = PrepareImageParams(dst_size);

        % [1] Template for 10x10 microneedle array (four 3x3 corners for orientation)
        [blank, mask] = MakeBlankNeedle(needle_array_size, corner_size, corner_pos, corner_hole, allow_inner);
        idx = find(mask > 0);

        needle = blank; 

        start_num = start_num_all(iratio);
        if save_single_file
            images_rect_ratio = zeros([num_images_all(iratio) dst_size], "uint8");
            labels_ratio = zeros([num_images_all(iratio) needle_array_size], "logical");

            images_raw_ratio = zeros([num_images_all(iratio) rect_raw_size], "uint8");
            corners_ratio = zeros([num_images_all(iratio) 4 2], "double");

            images_bin_ratio = zeros([num_images_all(iratio) rect_raw_size], "logical");
        end
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

            fprintf('  Image #%3d: %3d / %3d dots; max %3d, min %3d.\n', k, num_dots, length(idx), max(image(:)), min(image(:)));

            if save_single_file
                images_rect_ratio(i,:,:) = image;
                labels_ratio(i,:,:) = needle;

                images_raw_ratio(i,:,:) = imresize(uint8(image_raw*255), rect_raw_size);
                corners_ratio(i,:,:) = corner;

                images_bin_ratio(i,:,:) = imresize(image_bin, rect_raw_size);
            else
                imagefile = sprintf('%s/%03d_%06d.png', output_image_folder, needle_on_ratio*100, k);
                imwrite(image, imagefile);
                maskfile = sprintf('%s/%03d_%06d_mask.png', output_mask_folder, needle_on_ratio*100, k);
                imwrite(needle, maskfile);

                if save_figs
                    fig = figure('position',[50 50 600 250],'visible', 'off');
                    subplot(121);
                    imagesc(image);
                    colormap(gray);
                    title(sprintf('Simulated needle image #%d (input)',k));
                    subplot(122);
                    imagesc(needle);
                    colormap(gray);
                    title(sprintf('Needle array #%d, %d%% ON (label/target output)',k,needle_on_ratio*100));
                    saveas(fig,sprintf('%s/%03d_%06d_fig.png', output_fig_folder, needle_on_ratio*100, k));
                    close(fig);
                end
            end
        end

        if save_single_file
            images_rect(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = images_rect_ratio;
            labels(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = labels_ratio;

            images_raw(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = images_raw_ratio;
            corners(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = corners_ratio;

            images_bin(num_images_stepwise(iratio)+1:num_images_stepwise(iratio+1),:,:) = images_bin_ratio;
        end
    end
end

% save single .mat file in '-v7.3' version (>2GB)
if save_single_file
    save(recog_output_file, 'images_rect', 'labels', '-v7.3');
    save(rect_output_file, 'images_raw', 'corners', '-v7.3');
    save(bin_output_file, 'images_raw', 'images_bin', '-v7.3');
end

delete(gcp('nocreate'));
