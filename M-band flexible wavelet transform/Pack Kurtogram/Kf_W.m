function [Kf,M4,M2,k] = Kf_W(x,Nfft,Noverlap,Window,opt)
% [Kf,M4,M2] = Kf_W(x,Nfft,Noverlap,Window) 
% Welch's estimate of :       
%       1) the freq.-conditionned Kurtosis :  Kf(f) = M4(f)/M2(f)^2 - 2 
%       2) the 4th-order moment spectrum :   M4(f) = E{|X(f)|^4}
%       3) the 2nd-order moment spectrum :   M2(f) = E{|X(f)|^2}
%
% Caution : this version applies to stationary signals only !!此角本只适用于平稳信号
%
% x and y are divided into overlapping sections (Noverlap taps), each of which is
% detrended, windowed and zero-padded to length Nfft. 
% Note : use analytic signal to avoid correlation between + and -
% frequencies解析信号避免了正负频率的关系
% -----
%
% --------------------------
% Author: J. Antoni, 11-2003
% --------------------------

Window = Window(:)/norm(Window);		% Window Normalization归一化窗
n = length(x);								% Number of data points
nwind = length(Window); 				% length of window
if nwind<=Noverlap,
   error('nwind must be > Noverlap');%窗长必须大于重叠长度
end
x = x(:);		
k = fix((n-Noverlap)/(nwind-Noverlap));	% Number of windows每层分段数

% 1) Moment-based spectrum
% -------------------------
index = 1:nwind;
f = (0:Nfft-1)/Nfft;
t = (0:n-1)';
M4 = 0;
M2 = 0;

for i=1:k
   xw = Window.*x(index);%各段数据与窗相乘
   Xw = fft(xw,Nfft);%各段的傅利叶变换	
   if strcmp(opt,'kurt2')
      M4 = abs(Xw).^4 + M4;%各段四次统计量累加   
      M2 = abs(Xw).^2 + M2;%各段二次统计量累加
   else
      M4 = abs(Xw).^2 + M4;   
      M2 = abs(Xw) + M2;
   end
   index = index + (nwind - Noverlap);%下一个窗数据的下标
end

% normalize
M4 = M4/k;%四次统计量平均   
M2 = M2/k;%二次统计量累加 
Kf = M4./M2.^2;%本层分段峭度向量

if strcmp(opt,'kurt2')
   Kf = Kf - 2;
   b = 1;
else
   Kf = Kf - 1.27;
   b = .3;
end

% reduce biais near f = 0 mod(1/2)
W = abs(fft(Window.^2,Nfft)).^2;
Wb = zeros(Nfft,1);
for i = 0:Nfft-1,
   Wb(1+i) = W(1+mod(2*i,Nfft))/W(1);%下标为奇数的除以W（1）
end;
Kf = Kf - b*Wb;
