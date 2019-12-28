%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level)
% c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level)
% Calculates the kurtosis K of the complete quinte wavelet packet transform
% w of signal x, 计算五层小波包分解的峭度
% up to nlevel, using the lowpass and highpass filters h and g,
% respectively. 直到最后一层，分别用低通与高通滤波器
% The WP coefficients are sorted according to the frequency
% decomposition.小波包系数根据频率分解分类
% This version handles both real and analytical filters, but does not yiels WP coefficients
% suitable for signal synthesis.这个角本处理实信号与解析信号滤波，但不适用于合成信号的小波包系数
%
% -----------------------
% J閞鬽e Antoni : 12/2004 
% -----------------------   

nlevel = length(acoeff);
L = floor(log2(length(x)));
if nargin == 8
   if nlevel >= L
      error('nlevel must be smaller !!');
   end
   level = nlevel;
end
x = x(:);										 % shapes the signal as column vector if necessary

if nlevel == 0
   if isempty(bcoeff)
      c = x;
   else
      [c1,c2,c3] = TBFB(x,h1,h2,h3);
      if bcoeff == 0;
         c = c1(length(h1):end);
      elseif bcoeff == 1;
         c = c2(length(h2):end);
      elseif bcoeff == 2;
         c = c3(length(h3):end);
      end
   end
else
   c = K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
