function y = soft(x, T)
% y = soft(x, T)
%
% SOFT THRESHOLDING
% for real or complex data.
%
% INPUT
%   x : data (scalar or multidimensional array)
%   T : threshold (scalar or multidimensional array)
%
% OUTPUT
%   y : output of soft thresholding
%
% If x and T are both multidimensional, then they must be of the same size.

y = max(1 - T./abs(x), 0) .* x;
