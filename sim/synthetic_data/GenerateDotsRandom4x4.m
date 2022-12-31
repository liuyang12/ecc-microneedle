%GENERATEDOTSRANDOM10x10 Generate 10x10 randomly distributed microneedle
%array according to the template (four 3x3 corners for orientation)

clear; clc;
close all

% [0.0] directory configuration
addpath('../augment_image')
addpath('../prepare_image')

% [0.1] parameter configuration
num_images = 200; % Number of images to generate per pattern.
save_files = true; % save images or show them in figure window one-by-one
save_figs = true;  % save figures

dst_size = [48, 48]; % target image pixel resolution

needle_array_size = [4, 4]; % microneedle array sizeblank = zeros(needle_array_size, 'logical');
corner_size = 1; % p - corner size pxp for orientation
corner_pos = 'southeast'; % position of the pxp orientation pixels 
corner_hole = false; % has hole in the corner pixels or not

needle_on_ratio_all = [1.0, 0.75, 0.5, 0.25];

for iratio = 1:length(needle_on_ratio_all)
    needle_on_ratio = needle_on_ratio_all(iratio); % ratio of on pixels 
    % needle_on_ratio = 0.5; % ratio of on pixels 
    
    output_folder = sprintf('../output/needle%dx%d_image%dx%d_ratio%.2f',needle_array_size(2),needle_array_size(1),dst_size(2),dst_size(1),needle_on_ratio);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    if save_figs
        output_fig_folder = sprintf('../output/needle%dx%d_image%dx%d_ratio%.2f_fig',needle_array_size(2),needle_array_size(1),dst_size(2),dst_size(1),needle_on_ratio);
        if ~exist(output_fig_folder, 'dir')
            mkdir(output_fig_folder);
        end
    end

    dot_params = DotImageParams();
    augment_params = AugmentImageParams();
    prepare_params = PrepareImageParams(dst_size);

    % [1] Template for 10x10 microneedle array (four 3x3 corners for orientation)
    [blank, mask] = MakeBlankNeedle(needle_array_size, corner_size, corner_pos, corner_hole);
    idx = find(mask > 0);

    needle = blank; 

    % [2] Generate random micrneedle array according to the template
    for i = 1:num_images
        randvec = rand(size(idx))>1-needle_on_ratio;
        needle(idx) = randvec;
        
        % add random rotation 90 degrees to needle
        needle = rot90(needle, randi(4));
        
        [image, num_dots] = DotImage(needle, dot_params);
        image = AugmentImage(image, augment_params);
        % image = PrepareImage(image, num_dots); % [missing file] -> seem doing image resizing only
        image = prepareImage(image, prepare_params);

        fprintf('  Image #%d: %d / %d dots; max %d, min %d.\n', i, num_dots, length(idx), max(image(:)), min(image(:)));

        if (save_files)
            filename = sprintf('%s/image_%05d.png', output_folder, i);
            imwrite(image, filename);
            
            if save_figs
                fig = figure('position',[50 50 600 250],'visible', 'off');
                subplot(121);
                imagesc(image);
                colormap(gray);
                title(sprintf('Simulated needle image #%d (input)',i));
                subplot(122);
                imagesc(needle);
                colormap(gray);
                title(sprintf('Needle array #%d, %d%% ON (label/target output)',i,needle_on_ratio*100));
                saveas(fig,sprintf('%s/fig_%05d.png', output_fig_folder, i));
                close(fig);
            end
        else
            figure(1);
            imagesc(image);
            waitforbuttonpress;
        end
    end
end
    