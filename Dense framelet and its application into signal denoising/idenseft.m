function y =  idenseft(w, J, sf);
% INPUT
%   w : framelet(wavelet) coefficients
%   J : number of levels
%  sf : synthesis filter bank
% OUTPUT
%   y : reconstructed signal

y = w{J+1};
for j = J:-1:1
    y = sfb(y, w{j}{1}, w{j}{2}, sf);
end