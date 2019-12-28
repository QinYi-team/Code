% =========================================================================
%                          Written by Yi Qin
% =========================================================================
clc;
clear;
close all

%% generate the simulation signal
N= 6000;                                % M : signal length
fs= 1500;
t = (0:N-1)/fs;
ms=150;
x1=zeros(1,N);
s1=zeros(1,N);
while ms<N
x1(1:ms-1)=0;
x1(ms:N) = exp(-226.2*(t(ms:N)-ms/fs)).*sin(2*pi*720*(t(ms:N)-ms/fs)-0*pi);
cc1=1/sqrt(sum(x1.^2));
x1=7*cc1*x1;
s1=s1+x1;
ms=ms+700;
end
x1=s1;
x2 =0.5*cos(2*pi*350*t);
x3 =1.0*cos(2*pi*175*t);
yy= x1+x2+x3;
y=yy+1.5*randn(1,N);

%% Extract the transient signal
totalscal=1000;
x=impulseextraction(y,totalscal);

%% Output the result
figure(1)
subplot(211)
plot(t,y)
xlabel('\fontsize{11}\fontname{Times New Roman}\itt\rm\bf / \rms');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A')
subplot(212)
plot(t,x)
xlabel('\fontsize{11}\fontname{Times New Roman}\itt\rm\bf / \rms');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A')