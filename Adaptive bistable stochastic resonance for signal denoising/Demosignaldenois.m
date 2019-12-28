% =========================================================================
%                          Written by Yi Qin
% =========================================================================
clc
clear
close all
N=1024*2;
fs=2000;
f0=8;
t=(0:N-1)/fs;
t=t';
s=1*sin(2*pi*f0*t)+0.6*sin(4*pi*f0*t);

ns=1.9*randn(1,N)';
x=s+ns;
ratio=400;
Y=AdaptiveSR(x,fs,ratio);
figure(1)
plot(t,Y)
figure(2)
aa=abs(fft(Y));
df = fs/N;               %∆µ”Ú∑÷±Ê¬  Hz
ff1 = (0:floor(N/2)-1)*df;      %∆µ”Ú–Ú¡–
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;
plot(ff1,aa1,'-b')