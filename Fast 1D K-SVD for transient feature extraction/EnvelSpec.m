function [f1,a1]=EnvelSpec(s,fs,m)
% =========================================================================
%                          Written by Yi Qin
% =========================================================================
a=abs(fft(abs(hilbert(s))));
N=length(a);
df = fs/N;               %频域分辨率 Hz
f1 = (0:floor(N/2)-1)*df;      %频域序列
a1= 2*a(1:floor(N/2))/N;
a1(1)=a1(1)/2;          %零频分量不乘2
a1(1)=0;
a2=a1;
plot(f1,a1,'linewidth',2);
xlim([0 m])
% title('Envelope Spectrum')