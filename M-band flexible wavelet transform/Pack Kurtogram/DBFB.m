function [a,d] = DBFB(x,h,g)%低通滤波并降采样后信号存储在a中，高通滤波并降采样后信号存储在d中
% Double-band filter-bank.二分滤波器组
%   [a,d] = DBFB(x,h,g) computes the approximation
%   coefficients vector a and detail coefficients vector d,计算大约系数a及细节系数d
%   obtained by passing signal x though a two-band analysis
%   filter-bank.通过信号的二分带通滤波器得到
%   h is the decomposition low-pass filter and低通滤波器
%   g is the decomposition high-pass filter.高能滤波器

N = length(x);
La = length(h);
Ld = length(g);

% lowpass filter
a = filter(h,1,x);%低通滤波
a = a(2:2:N);%降采样
a = a(:);

% highpass filter
d = filter(g,1,x);%高通滤波
d = d(2:2:N);%降采样
d = d(:);%降采样

% ------------------------------------------------------------------------