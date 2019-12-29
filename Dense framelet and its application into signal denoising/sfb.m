function y = sfb(lo, bp, hi, sf)

% SYNTHESIS FILTER BANK
%    with cyclic convolution
% y = sfb(lo, bp, hi, sf)
% INPUT
%   lo : lowpass subband signal
%   bp : bandpass subband signal
%   hi : highpass subband signal
%   sf : synthesis filters
% OUTPUT
%   y  : output signal

g0 = sf(:,1);    % lowpass filter
g1 = sf(:,2);    % bandpass filter
g2 = sf(:,3);    % highpass filter

% length of output signal
L = 2*length(lo);

% upsample and filter lowpass subband signal
lo = up(lo,2);
lo = conv(lo,g0);

% upsample and filter bandpass subband signal
bp = up(bp,2);
bp = conv(bp,g1);

% filter highpass subband signal
hi = conv(hi,g2);

% add signals
if length(lo)<length(hi)
    lo=[lo 0];
    bp=[bp 0];
end
y = lo + bp + hi;

% wrap the tail to the front.
k = 1:(length(y)-L);
y(k) = y(k) + y(L+k);
y = y(1:L);

% perform circular shift to remove delay
N = length(g0);
k = mod([0:L-1]+N-1,L)+1;
y = y(k);
