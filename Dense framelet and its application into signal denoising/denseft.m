function w = denseft(x, J, af);
% INPUT
%   x  : input signal
%   J  : number of levels
%   af : analysis filter bank
% OUTPUT
%   w  : framelet(wavelet) coefficients

for j = 1:J
    [x, w{j}{1}, w{j}{2}] = afb(x, af);
end
w{J+1} = x;