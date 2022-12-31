function [blank, mask] = MakeBlankNeedle(needle_array_size, corner_size, corner_pos, corner_hole, allow_inner)
%MAKEBLANKNEEDLE Make a blank microneedle array.

if nargin < 1, needle_array_size = [10,10]; end
if nargin < 2, corner_size = 3; end
if nargin < 3, corner_pos = 'southeast'; end
if nargin < 4, corner_hole = true; end
if nargin < 4, allow_inner = false; end

blank = zeros(needle_array_size, 'logical');

p = corner_size; % corner pxp pixels for orientation detection

blank(1:p,1:p) = 1;
blank(1:p,end-p+1:end) = 1;
blank(end-p+1:end,1:p) = 1;
blank(end-p+1:end,end-p+1:end) = 1;

mask = 1 - blank;

pp = floor(p/2);

blank(1:p,1:p) = 0;

if corner_hole
    blank(pp+1,end-pp) = 0;
    blank(end-pp,pp+1) = 0;
    blank(end-pp,end-pp) = 0;
end

if allow_inner
    blank(p, p) = 0;
    blank(end-p+1,p) = 0;
    blank(p, end-p+1) = 0;
end

switch lower(corner_pos)
    case {'northwest','north-west','nw'}
        rot_time = 0;
    case {'southwest','south-west','sw'}
        rot_time = 1;
    case {'southeast','south-east','se'}
        rot_time = 2;
    case {'northeast','north-east','ne'}
        rot_time = 3;
end

blank = rot90(blank, rot_time);

end

