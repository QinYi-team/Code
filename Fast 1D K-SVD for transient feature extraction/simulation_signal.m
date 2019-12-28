% =========================================================================
%                          Written by Yi Qin
% =========================================================================
function [s] = simulation_signal(N,t)
%生成仿真信号

s1 = 0.4*(cos(2*pi*3.5*t) + 2*cos(2*pi*35*t) + cos(2*pi*5*t).*cos(2*pi*35*t) + cos(2*pi*25*t+2*sin(2*pi*t)));
ms=50;
s2=zeros(1,N);
% fc = 4529.65/(2*pi);% 中心频率2Hz 
% ce=0.05;                                    %生成脉冲信号,初始为0.05
while ms<N
x1(1:ms)=0;                                 %生成脉冲信号
% x1(ms:N) = exp(-2*pi*fc*ce*(t(ms:N)-ms/fs)).*sin(2*pi*fc*sqrt(1-ce^2)*(t(ms:N)-ms/fs)-0*pi);
% % x1(ms:M) = exp(-0.05/sqrt(0.9975)*2*pi*3000*(m(ms:M)-ms/fs));
x1(ms:N) = exp(-240*t(ms:N)).*cos(500*pi*t(ms:N));
cc1=1/sqrt(sum(x1.^2));
x1=1.5*cc1*x1;
s2 = s2 + x1;
ms = ms + 100;
end
s2=2*s2;
n=0.25*randn(1,N);

s = s1'+ s2'+n';        %仿真信号