function [J, corners_raw] = imageCrop(I, params, warp, corners, I_bin)

addpath('../image_process')

corners = corners .* size(I); % relative to absolute
corners_aug = transformPointsForward(warp, corners);
corners_raw = corners_aug ./ size(I);

switch lower(params.crop_method)
    case 'homography' % rectification based on ground truth homography (four corners with minor shift errors)        
        % corners = corners .* size(I); % relative to absolute
        % corners_aug = transformPointsForward(warp, corners);
        % corners_raw = corners_aug ./ size(I);
        
        corners_aug_shift = corners_aug + rand(size(corners_aug))*max(params.corner_shift_range); % homography with minor errors on four corners

        h = params.dst_size(1);
        w = params.dst_size(2);
        rect_pts = [1 1; w 1; w h; 1 h];
        rect_tform = fitgeotrans(corners_aug_shift, rect_pts, 'projective');
        J = imwarp(I, rect_tform, 'OutputView', imref2d(params.dst_size));
        
        if params.fig_show
            % plot the corners of images before and after warping
            h = figure(1);
            h.Position = [50 50 800 800]; 
            subplot(221); imshow(I); drawpolygon('Position',corners_aug); drawpolygon('Position',corners_aug_shift,'Color','m'); title('warpped image');
            subplot(222); imshow(J); title('warpped image after rectification');

            I_sz = imresize(I, params.dst_size);
            corners_aug = corners_aug./size(I).*size(I_sz);
            corners_aug_shift = corners_aug_shift./size(I).*size(I_sz);
            rect_tform = fitgeotrans(corners_aug_shift, rect_pts, 'projective');
            J_sz = imwarp(I_sz, rect_tform, 'OutputView', imref2d(params.dst_size));

            subplot(223); imshow(I_sz); drawpolygon('Position',corners_aug); drawpolygon('Position',corners_aug_shift,'Color','m'); title('warpped image (dst size)');
            subplot(224); imshow(J_sz); title('warpped image after rectification (dst size)');
        end
        
    case 'naive' % naive cropping the rectangle of four corners
        % binary thresholding the image
        I_filter = imgaussfilt(I, params.filter_sigma);
        threshold = max(I_filter(:)) / 2;
        I_thresh = I_filter > threshold;
        
        [i, j] = find(I_thresh);
        i0 = min(i);
        i1 = max(i);
        j0 = min(j);
        j1 = max(j);

        ic = round((i0 + i1) / 2);
        jc = round((j0 + j1) / 2);
        sz = 0.5 * (max(i1 - i0, j1 - j0) + 1);
        sz = sz * (1 + params.crop_margin);
        sz = round(sz);

        % Make sure the crop fits within image bounds (trim to the bounds).
        imin = max(1, ic - sz);
        jmin = max(1, jc - sz);
        imax = min(ic + sz, size(I, 1));
        jmax = min(jc + sz, size(I, 2));
        szmin = min(imax - imin, jmax - jmin);
        imax = imin + szmin;
        jmax = jmin + szmin;

        J = I(imin:imax, jmin:jmax, :);
    case 'minrect'
        % binary thresholding the image
%         I_filter = imgaussfilt(I, filter_sigma);
%         threshold = max(I_filter(:)) / 2;
%         I_thresh = I_filter > threshold;
        % I_bin = imwarp(I_bin, warp, 'OutputView', imref2d(size(I_bin)));
        I_thresh = I_bin > 0;
        
        [i, j] = find(I_thresh);
        [bbox, angle] = minBoundingBox([i,j]');
        
        center = mean(bbox, 2); % 2x4 -> 2x1
        
        sz = sqrt(sum(abs(bbox(:,[2,4]) - bbox(:,1)).^2,1));
        sz = sz*(1+params.crop_margin);
        
        
        
        % rot_angle = min(angle,pi/2-angle);
        if angle < pi/4
            rot_angle = -angle;
        else
            rot_angle = pi/2-angle;
            sz = sz([2,1]);
        end
        
        rect = round([center([2,1])'-sz/2, sz]);
        
        rotMat = getRotationMatrix2D(center, rot_angle, 1);
        
        tform = affine2d(rotMat);
        
        sameAsInput = affineOutputView(size(I),tform,'BoundsStyle','SameAsInput');
        I_rot = imwarp(I, tform, 'OutputView', sameAsInput);
        
        J = imcrop(I_rot, rect);
        
        J = imresize(J, params.dst_size);
        
        if params.fig_show
            % figure(3);
            % plot(i,j,'.'); hold on
            % plot(bbox(1,[1:end 1]),bbox(2,[1:end 1]),'r');
            % axis equal;
        
            h = figure(1);
            h.Position = [50 50 800 800]; 
            subplot(221); imshow(I); title('raw image');
            subplot(222); imshow(I_bin>0); title('binairzed image');
            subplot(223); imshow(I_rot); title('rotated image')
            subplot(224); imshow(J); title('cropped image');
        end
        
    otherwise
        error('Unsupported cropping method %s!',crop_method);
end
