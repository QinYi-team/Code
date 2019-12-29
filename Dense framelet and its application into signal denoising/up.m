function y = up(x,M)
% y = up(x,M)
% M-fold up-sampling of a 1-D signal

[r,c] = size(x);
if r > c
   y = zeros(M*r,1);
else
   y = zeros(1,M*c);
end
y(1:M:end) = x;

