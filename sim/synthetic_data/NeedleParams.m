function params = NeedleParams( needle_side_size )
% Needle parameters according to the side size of the needle patch

switch needle_side_size
    case 10
        params.needle_array_size = [needle_side_size, needle_side_size]; % microneedle array size
        params.rect_raw_size = [256, 256]; % raw image size for rectification
        params.corner_size = 3; % p - corner size pxp for orientation
        params.corner_pos = 'southeast'; % position of the pxp orientation pixels 
        params.corner_hole = true; % has hole in the corner pixels or not
        params.allow_inner = false; % allow innner pixels
        
    case 12
        params.needle_array_size = [needle_side_size, needle_side_size]; % microneedle array size
        params.rect_raw_size = [256, 256]; % raw image size for rectification
        params.corner_size = 2; % p - corner size pxp for orientation
        params.corner_pos = 'southeast'; % position of the pxp orientation pixels 
        params.corner_hole = false; % has hole in the corner pixels or not
        params.allow_inner = false; % allow innner pixels
      
    case 17
        params.needle_array_size = [needle_side_size, needle_side_size]; % microneedle array size
        params.rect_raw_size = [256, 256]; % raw image size for rectification
        params.corner_size = 3; % p - corner size pxp for orientation
        params.corner_pos = 'southeast'; % position of the pxp orientation pixels 
        params.corner_hole = true; % has hole in the corner pixels or not
        params.allow_inner = true; % allow innner pixels
        
    otherwise 
        params.needle_array_size = [needle_side_size, needle_side_size]; % microneedle array size
        params.rect_raw_size = [256, 256]; % raw image size for rectification
        params.corner_size = 2; % p - corner size pxp for orientation
        params.corner_pos = 'southeast'; % position of the pxp orientation pixels 
        params.corner_hole = false; % has hole in the corner pixels or not
        params.allow_inner = false; % allow innner pixels
        
end

params.dst_size = ((params.needle_array_size+4)*2+2)*4; % target image pixel resolution


