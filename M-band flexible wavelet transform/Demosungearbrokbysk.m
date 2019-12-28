% =========================================================================
%                          Written by Yi Qin
% =========================================================================
clc
clear
close all

load sungearbroken    %% input speed 40Hz, the sun gear has a broken tooth, and the fault characteristic frequency is 5.2Hz
fs= 12800;   %% sampling frequency
N=length(x);
t=(0:N-1)/fs;

hl=figure(1);
p_vect=[700 400 660 540];
set(hl,'Position',p_vect);
%% plot original signal
subplot(311)
plot(t,x)
xlabel('\fontsize{11}\fontname{Times New Roman}\itt\rm\bf / \rms');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');

%% plot frequency spectrum
subplot(312)
aa=abs(fft(x));

df = fs/N;               
ff1 = (0:floor(N/2)-1)*df;      
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontsize{11}\fontname{Times New Roman}\it f\rm\bf / \rmHz');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');

%% plot envelope spectrum
subplot(313)
aa=abs(fft(abs(hilbert(x))));
N=length(aa);
df = fs/N;               
ff1 = (0:floor(N/2)-1)*df;      
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontsize{11}\fontname{Times New Roman}\it f\rm\bf / \rmHz');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');
xlim([0 30])


c = Fast_Kurtogram(x,6,fs);   
ccc=resample(real(c),length(x),length(c));
hl=figure(4);
p_vect=[700 400 660 400];
set(hl,'Position',p_vect);
subplot(211)
 plot(t,real(ccc))
 xlim([0 max(t)])
xlabel('\fontsize{11}\fontname{Times New Roman}\itt\rm\bf / \rms');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');

subplot(212)
env=abs(hilbert(ccc));
aa=abs(fft(env));
N=length(aa);
df = fs/length(x);               
ff1 = (0:floor(N/2)-1)*df;      
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;        
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontsize{11}\fontname{Times New Roman}\it f\rm\bf {/} \rmHz');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');
xlim([0 30])