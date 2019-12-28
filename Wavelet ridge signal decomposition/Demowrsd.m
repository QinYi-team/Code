% =========================================================================
%                          Written by Yi Qin
% =========================================================================
clc;
clear all;
close all;
N=2048;
fs=1000;
t=(0:N-1)/fs;
%%%%%%%%%%%chirp signal
% y=sin(2*pi*5*(t.^2+1*t))+0*randn(1,N);
%%%%%%%%%%%multicomponent harmonic signal
y4=0.9*cos(2*pi*90*t-pi/6);
y3=1*cos(2*pi*60*t+pi/4);
y2=1.4*cos(2*pi*30*t+pi/3);
y1=1.0*cos(2*pi*15*t);
y=1*y1+y2+y3+1*y4;
%%%%%%%%%%%Parameter initialization
a0=fs/90;
lf=0.02;
n=300;
fb=2;
fc=1;
num=4; %%the  number of target components
D=WRSD(y,a0,n,fb,fc,fs,lf,num);
figure(1)
subplot(411)
plot(t,D(1,:))
xlim([t(1) t(end)])
subplot(412)
plot(t,D(2,:))
xlim([t(1) t(end)])
subplot(413)
plot(t,D(3,:))
xlim([t(1) t(end)])
subplot(414)
plot(t,D(4,:))
xlim([t(1) t(end)])
