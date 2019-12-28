%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xf,Nw,fc] = Find_stft_kurt(x,nlevel,LNw,Fr,opt,Fs)
% [xf,Nw,fc] = Find_stft_kurt(x,nlevel,LNw,Fr,opt2)
% LNw = log2(Nw) with Nw the analysis window of the stft
% Fr is in [0 .5]
%
% -------------------
% J. Antoni : 12/2004
% -------------------

if nargin < 6
   Fs = 1;
end

Nfft = 2.^[3:nlevel+2];%每层分段数，只包含二分层					
temp = [3*Nfft(1)/2 3*Nfft(1:end-2);Nfft(2:end)];
Nfft = [Nfft(1) temp(:)'];%每层分段数，包括三分层
LNw_stft = [0 log2(Nfft)];%每层的纵坐标，包括未分解前信号			
[temp,I] = min(abs(LNw_stft-LNw));%找到滤波层及其行坐标
Nw = 2^LNw_stft(I);%滤波层分段数

NFFT = 2^nextpow2(Nw);%滤波层的傅利叶变换长度
freq_stft = (0:NFFT/2-1)/NFFT;%傅利叶变换后的频率采样点
[temp,J] = min(abs(freq_stft-Fr+1/NFFT/4));%查找最接近载波频率的频率位置
fc = freq_stft(J);%滤波带中心频率位置

if LNw > 0
   b = hanning(Nw)';%滤波层窗
   b = b/sum(b);%归一化
   b = b.*exp(2i*pi*(0:Nw-1)*fc);%滤波器
   xf = fftfilt(b,x);%滤波
   xf = xf(fix(Nw/2)+1:end);
   dt = fix(Nw/4);					% downsample by at least 4 samples per window (this corresponds to 75% overlap)
else
   xf = x;
   Nw = 0;
   dt = 1;
end

b = hanning(Nw)';%滤波层窗
   b = b/sum(b);%归一化
   b = b.*exp(2i*pi*(0:Nw-1)*fc);%滤波器
   xf = fftfilt(b,x);%滤波
   xf = xf(fix(Nw/2)+1:end);
   dt = fix(Nw/4);


env = abs(xf(dt:dt:end)).^2;%滤波后信号包络平方

%temp = xf.*exp(-2i*pi*(0:length(xf)-1)'*fc);
%figure,subplot(211),plot(real(temp))
%subplot(212),plot(real(temp(dt:dt:end)))
kx = kurt(xf(dt:dt:end),opt);%滤波后信号峭度
sig = median(abs(xf(dt:dt:end)))/sqrt(pi/2);%median取均值sqrt求平方根
threshold = sig*raylinv(.999,1);%阈值

spec = input('	Do you want to see the envelope spectrum (yes = 1 ; no = 0): ');
figure
t = (0:length(x)-1)/Fs;%原始信号采样时间点
tf = t(fix(Nw/2)+1:end);%滤波信号采样点
subplot(2+spec,1,1),plot(t,x,'k'),title('Original signal')%原始信号时域波形图
subplot(2+spec,1,2),plot(tf,real(xf),'k'),hold on,plot(tf,threshold*ones(size(xf)),':r'),plot(tf,-threshold*ones(size(xf)),':r'),%plot(abs(xf),'r'),
title(['Filtered signal, Nw=2^{',num2str(LNw_stft(I)),'}, fc=',num2str(Fs*fc),'Hz, Kurt=',num2str(fix(10*kx)/10),', \alpha=.1%'])
xlabel('time [s]')
if spec == 1
   nfft = 2^nextpow2(length(env));
   S = abs(fft((env(:)-mean(env)).*hanning(length(env)),nfft)/length(env));
   f = linspace(0,.5*Fs/dt,nfft/2);
   subplot(3,1,3),plot(f,S(1:nfft/2),'k'),title('Fourier transform magnitude of the squared filtered signal'),xlabel('frequency [Hz]')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%