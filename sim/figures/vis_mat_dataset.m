% clear; clc; 
% close all

% load saved .mat dataset

% load('/data/yliu/proj/microneedle/data/needle10x10_image120x120_ord_100k_recog.mat')
% load('/data/yliu/proj/microneedle/data/needle10x10_image120x120_ord_100k_rect.mat')

% load('/data/yliu/proj/microneedle/data/needle10x10_image120x120_dim_50k_recog.mat');
% load('/data/yliu/proj/microneedle/data/needle10x10_image120x120_dim_50k_rect.mat')

% load('../output/needle10x10_1.000000e-01k_recog.mat')
% load('../output/needle10x10_1.000000e-01k_rect.mat')
% load('../output/needle10x10_1.000000e-01k_bin.mat')

% load('/data/yliu/proj/microneedle/data/needle17x17_1k_recog.mat')
% load('/data/yliu/proj/microneedle/data/needle17x17_1k_rect.mat')
% load('/data/yliu/proj/microneedle/data/needle17x17_1k_bin.mat')

load('/data/yliu/proj/microneedle/data/needle10x10_100k_recog.mat')
load('/data/yliu/proj/microneedle/data/needle10x10_100k_rect.mat')
load('/data/yliu/proj/microneedle/data/needle10x10_100k_bin.mat')

%% access individual image/label data for visualization
idx = 90000;
idx = 100;

im_rect = squeeze(images_rect(idx,:,:));
label = squeeze(labels(idx,:,:));

im_raw = squeeze(images_raw(idx,:,:));
corner = squeeze(corners(idx,:,:));

im_bin = squeeze(images_bin(idx,:,:));

dst_size = size(im_raw);
corner_raw = corner .* dst_size;

% plot raw images and labels
f = figure(1);
f.Position = [100,100,1400,1000];

h = dst_size(1);
w = dst_size(2);
rect_pts = [1 1; w 1; w h; 1 h];
rect_tform = fitgeotrans(corner_raw, rect_pts, 'projective');
im_raw_rect = imwarp(im_raw, rect_tform, 'OutputView', imref2d(dst_size));

subplot(231); 
  imagesc(im_raw); axis image; colormap gray; 
  drawpolygon('Position', corner_raw);
  title('raw image (downsampled)');

subplot(232); 
  imagesc(im_bin); axis image; colormap gray; 
  drawpolygon('Position', corner_raw);
  title('binarized image (downsampled)');
  
subplot(234);
  imagesc(im_raw_rect); axis image; colormap gray; 
  title('rectified raw image (downsampled)');

subplot(235); 
  imagesc(im_rect); axis image; colormap gray; title('rectified image');
subplot(236); 
  imagesc(label); axis image; colormap gray; title('microneedle array (binary)');
