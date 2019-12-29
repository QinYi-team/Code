% =========================================================================
%                          Written by Yi Qin
% =========================================================================
close all
clc;
clear;

load gearnois  %%noisy gear signal, and the fault characteristci frequency is 10.67Hz 
fs=6144;
N=length(x);
t=(0:N-1)/fs;
n=5; %decomposition level
sorh = 'h'; %%hard thresholding
%%%%%%%%%%%%%%Denoise by my spline framelet%%%%%%%%%%%%
load spline3
y1 = denseftdeno(x,af,n,sorh);

%%%%%%%%%%%%%%Denoise by Selesnick's wavelet%%%%%%%%%%%%
load HDDWT43
y2 = denseftdeno(x,af,n,sorh);

%%%%%%%%%%%%%%Denoise by Daubechies wavelet%%%%%%%%%%%%
wname='db3';
method = 'rigrsure'; 
scal = 'one';
y3 = wden(x,method,sorh,scal,n,wname);

%%%%%Original gear signal
hf_fig=figure(1);
p_vect=[440 400 500 420];
set(hf_fig,'Position',p_vect);
subplot(211)
plot(t,x)
xlabel('\fontname{Times New Roman}Time\fontname{Times New Roman}\it t\rm\bf / \rms');
ylabel('\fontname{Times New Roman}Acceleration\fontname{Times New Roman}\it  a\rm{ / (m }\bf{\cdot}\rm s^{-2})');
title('\fontname{Times New Roman}Original gear signal and its frequency spectrum')
%%%%%Frequency spectrum of original signal
subplot(212)
aa=abs(fft(x));
N=length(aa);
df =fs/N;               
ff1 = (0:floor(N/2)-1)*df;      
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;         
plot(ff1, aa1);
xlabel('\fontname{Times New Roman}Frequency\it \rmHz');
ylabel('\fontname{Times New Roman}Amplitude\rm{   m }\bf{\cdot}\rm s^{-2}');
xlim([0 max(ff1)])

%%%%%the denoised result obtained by the proposed spline framelet
hf_fig=figure(2);
set(hf_fig,'Position',p_vect);
subplot(211)
plot(t,y1)
xlabel('\fontname{Times New Roman}Time\fontname{Times New Roman}\it t\rm\bf / \rms');
ylabel('\fontname{Times New Roman}Acceleration\fontname{Times New Roman}\it  a\rm{ / (m }\bf{\cdot}\rm s^{-2})');
title('\fontname{Times New Roman}Denoised result obtained by the proposed spline framelet')
%%%%%Envelope spectrum of denoised result obtained by the proposed spline framelet
subplot(212)
N=length(y1);
aa=abs(fft(abs(hilbert(y1))));
df = fs/N;             
ff1 = (0:floor(N/2)-1)*df;  
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;        
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontname{Times New Roman}\fontsize{11}Frequency\it  \rmHz');
ylabel('\fontname{Times New Roman}Amplitude\rm{   m }\bf{\cdot}\rm s^{-2}');
xlim([0 120])

%%%%%the denoised result obtained by the Selesnick's wavelet
hf_fig=figure(3);
set(hf_fig,'Position',p_vect);
subplot(211)
plot(t,y2)
xlabel('\fontname{Times New Roman}Time\fontname{Times New Roman}\it t\rm\bf / \rms');
ylabel('\fontname{Times New Roman}Acceleration\fontname{Times New Roman}\it  a\rm{ / (m }\bf{\cdot}\rm s^{-2})');
title('\fontname{Times New Roman}Denoised result obtained by Selesnick wavelet')
%%%%%Envelope spectrum of denoised result obtained by the Selesnick's wavelet
subplot(212)
N=length(y2);
aa=abs(fft(abs(hilbert(y2))));
df = fs/N;             
ff1 = (0:floor(N/2)-1)*df;  
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;        
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontname{Times New Roman}\fontsize{11}Frequency\it  \rmHz');
ylabel('\fontname{Times New Roman}Amplitude\rm{   m }\bf{\cdot}\rm s^{-2}');
xlim([0 120])

%%%%%the denoised result obtained by Daubechies wavelet
hf_fig=figure(4);
set(hf_fig,'Position',p_vect);
subplot(211)
plot(t,y3)
xlabel('\fontname{Times New Roman}Time\fontname{Times New Roman}\it t\rm\bf / \rms');
ylabel('\fontname{Times New Roman}Acceleration\fontname{Times New Roman}\it  a\rm{ / (m }\bf{\cdot}\rm s^{-2})');
title('\fontname{Times New Roman}Denoised result obtained by Daubechies wavelet')
%%%%%Envelope spectrum of denoised result obtained by Daubechies wavelet
subplot(212)
N=length(y3);
aa=abs(fft(abs(hilbert(y3))));
df = fs/N;             
ff1 = (0:floor(N/2)-1)*df;  
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;        
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontname{Times New Roman}\fontsize{11}Frequency\it  \rmHz');
ylabel('\fontname{Times New Roman}Amplitude\rm{   m }\bf{\cdot}\rm s^{-2}');
xlim([0 120])
