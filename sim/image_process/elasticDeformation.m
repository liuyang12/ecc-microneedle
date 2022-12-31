function deformed = elasticDeformation(im)

% code credit: https://www.mathworks.com/matlabcentral/fileexchange/66663-elastic-distortion-transformation-on-an-image

% Compute a random displacement field
dx = -1 + 2*rand(size(im)); % dx ~ U(-1,1)
dy = -1 + 2*rand(size(im)); % dy ~ U(-1,1)

% Normalizing the field
nx = norm(dx);
ny = norm(dy);
dx = dx./nx; % Normalization: norm(dx) = 1
dy = dy./ny; % Normalization: norm(dy) = 1

% Smoothing the field
sig = 24; % Standard deviation of Gaussian convolution
alpha = 48; % Scaling factor
fdx = imgaussfilt(dx,sig,'FilterSize',7); % 2-D Gaussian filtering of dx
fdy = imgaussfilt(dy,sig,'FilterSize',7); % 2-D Gaussian filtering of dy

% Filter size: 2 * 3*ceil(std2(dx)) + 1
% = 3 sigma pixels in each direction + 1 to make an odd integer
fdx = alpha*fdx; % Scaling the filtered field
fdy = alpha*fdy; % Scaling the filtered field

% The resulting displacement
[y,x] = ndgrid(1:size(im,1),1:size(im,2));

figure(1)
imagesc(im); colormap gray; axis image; axis tight; hold on;
title('Displacement field')
quiver(x,y,fdx,fdy,0,'r')

% Applying the displacement to the original pixels
deformed = griddata(x-fdx,y-fdy,double(im),x,y);
deformed(isnan(deformed)) = 0;

end

