function [a1,a2,a3] = TBFB(x,h1,h2,h3)
% Trible-band filter-bank.三分滤波器组
%   [a1,a2,a3] = TBFB(x,h1,h2,h3) 

N = length(x);
La1 = length(h1);
La2 = length(h2);
La3 = length(h3);

% lowpass filter
a1 = filter(h1,1,x);%低通滤波
a1 = a1(3:3:N);%降采样
a1 = a1(:);

% passband filter
a2 = filter(h2,1,x);%带通滤波
a2 = a2(3:3:N);%降采样
a2 = a2(:);

% highpass filter
a3 = filter(h3,1,x);%高通滤波
a3 = a3(3:3:N);%降采样
a3 = a3(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%