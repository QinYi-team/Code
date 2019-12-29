function [lo, bp, hi] = afb(x, af)
% [lo, bp, hi] = afb(x, af)
% INPUT
%   x  : input signal (even-length)
%   af : analysis filter bank
% OUTPUT
%   lo : lowpass subband signal
%   bp : bandpass subband signal
%   hi : highpass subband signal

h0 = af(:,1);   % lowpass filter
h1 = af(:,2);   % bandpass filter
h2 = af(:,3);   % highpass filter

L = length(x);     % length of input signal

% --- lowpass channel ---
lo = conv(x,h0);   % filter with h0
lo = lo(1:2:end);  % downsample

% wrap the tail to the front.
k = 1:(length(lo)-L/2);
lo(k) = lo(k) + lo(L/2+k);
lo = lo(1:L/2);

% --- bandpass channel ---
bp = conv(x,h1);   % filter with h1
bp = bp(1:2:end);  % downsample

% wrap the tail to the front.
k = 1:(length(bp)-L/2);
bp(k) = bp(k) + bp(L/2+k);
bp = bp(1:L/2);


% --- highpass channel ---
hi = conv(x,h2);   % filter with h2

% wrap the tail to the front.
k = 1:(length(hi)-L);
hi(k) = hi(k) + hi(L+k);
hi = hi(1:L);

