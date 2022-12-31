function rotMat = getRotationMatrix2D(center, angle, scale)
%GETROTATIONMATRIX2D Get rotation matrix for 2D image transformation with
%rotation center, angle, and scale.

a = scale*cos(angle);
b = scale*sin(angle);

y = center(1);
x = center(2);

rotMat = [  a,  b, (1-a)*x-b*y;
           -b,  a, b*x+(1-a)*y; 
            0,  0,           1 ]';
      
end