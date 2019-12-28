% =========================================================================
%                          Written by Yi Qin
% =========================================================================
clc
clear
close all

fs=25600;   %%the sampling frequency is 107.8Hz
load bearoutfault  %%The bearing has a outer race fault,and the fault characteristic frequency is 107.8Hz
N=length(s);
t=(0:N-1)/fs;

%  Plot the original signal
hl=figure(1);
subplot(211)
plot(t,s)
xlabel('\fontsize{11}\fontname{Times New Roman}\itt\rm\bf / \rms');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');
xlim([t(1) t(end)])

%  Plot the envelope spectrum of the original signal
subplot(212)
aa=abs(fft(abs(hilbert(s))));
N=length(aa);
df = fs/N;            
ff1 = (0:floor(N/2)-1)*df;    
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;          
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontsize{11}\fontname{Times New Roman}\it f\rm\bf / \rmHz');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');
xlim([0 1000])

wn = 4487;
sigma=0.3;
fws=500; 
wavsupport=0.6;
scales=1:1:60;
rwc=0.009;

%  A2 denotes DFT and A2T denotes IDFT
M= N;                   
A2 = @(x) A(x,N,M)/sqrt(M);
A2T = @(x) AT(x,N,M)/sqrt(M);
p2 = 1;
p1=1;
 
theta = 0.6;              % trade-off parameter
itn =35;                  % number of iterations
mu1 = 0.005;              % mu2 should be properly set for different signlas 
mu2 = 1.0;

[y1,y2,cost]= itershrink(s, scales,wn,sigma,fws,wavsupport,rwc, p1, A2, A2T, p2, theta, 1-theta, mu1, mu2, itn);

%  Plot the extracted impulsive signal
figure(2)
subplot(211)
plot(t,y1)
xlabel('\fontsize{11}\fontname{Times New Roman}\itt\rm\bf / \rms');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');
xlim([t(1) t(end)])

%  Plot the envelope spectrum of the extracted impulsive signal
subplot(212)
aa=abs(fft(abs(hilbert(y1))));
N=length(aa);
df = fs/N;               
ff1 = (0:floor(N/2)-1)*df;      
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;         
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontsize{11}\fontname{Times New Roman}\it f\rm\bf / \rmHz');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');
xlim([0 1000])