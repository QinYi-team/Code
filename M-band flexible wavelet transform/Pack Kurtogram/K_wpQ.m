function K = K_wpQ(x,h,g,h1,h2,h3,nlevel,opt,level)
% K = K_wpQ(x,h,g,h1,h2,h3,nlevel)
% Calculates the kurtosis K of the complete quinte wavelet packet transform
% w of signal x, 计算信号x的五层小波包变换峭度
% up to nlevel, using the lowpass and highpass filters h and g, respectively.从上到最低层分别用低通高通滤波器 
% The WP coefficients are sorted according to the frequency
% decomposition.系数根据频率分解分类
% This version handles both real and analytical filters, but does not yiels WP coefficients
% suitable for signal synthesis.此角本适用于实信号及解析信号的滤波，不适用于合成信号
%
% -----------------------
% J閞鬽e Antoni : 12/2004 
% -----------------------   

L = floor(log2(length(x)));
if nargin == 8
   if nlevel >= L%判断分解层是否超出最大允许层
      error('nlevel must be smaller !!');
   end
   level = nlevel;
end
x = x(:);										 % shapes the signal as column vector if necessary

[KD,KQ] = K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level);%计算每层峭度，初始信号与二分层峭度存储于KD，三分层峭度存储于KQ）

K = zeros(2*nlevel,3*2^nlevel);%设定峭度的行列数，最低层的每一段由三列构成，行数为分解层数的二倍
K(1,:) = KD(1,:);%未分解前的峭度
for i = 1:nlevel-1
   K(2*i,:) = KD(i+1,:);%二分段层的峭度
   K(2*i+1,:) = KQ(i,:);%三分段层的峭度
end
K(2*nlevel,:) = KD(nlevel+1,:);%最后一层的峭度（二分)
