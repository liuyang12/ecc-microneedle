function R = RandomRotation(image_size, rot_degree)

if nargin < 2
    rot_degree = 180; % default rotation degree [-180°,180°]
end
if length(rot_degree) < 2
    rot_degree = [-rot_degree, rot_degree];
end

t = 2*pi * RandRange(rot_degree)/180;
c = cos(t);
s = sin(t);
R = [c -s 0; s c 0; 0 0 1];
center = image_size' / 2;
R(1:2, 3) = center - R(1:2, 1:2) * center;

end