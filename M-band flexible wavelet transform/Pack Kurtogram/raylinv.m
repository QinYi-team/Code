%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = raylinv(p,b)
%RAYLINV  Inverse of the Rayleigh cumulative distribution function
%(cdf).雷利累积分布函数
%   X = RAYLINV(P,B) returns the Rayleigh cumulative distribution 
%   function with parameter B at the probabilities in P.函数中的参数B概率为P

if nargin <  1, 
    error('Requires at least one input argument.'); 
end

% Initialize x to zero.
x = zeros(size(p));

% Return NaN if the arguments are outside their respective limits.
k1 = find(b <= 0| p < 0 | p > 1);
if any(k1) 
    tmp   = NaN;
    x(k1) = tmp(ones(size(k1)));
end

% Put in the correct values when P is 1.
k = find(p == 1);
if any(k)
    tmp  = Inf;
    x(k) = tmp(ones(size(k))); 
end

k=find(b > 0 & p > 0  &  p < 1);
if any(k),
    pk = p(k);
    bk = b(k);
    x(k) = sqrt((-2*bk .^ 2) .* log(1 - pk));
end