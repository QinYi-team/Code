
function wcoefs = myrmorletcwt(Sig,Scales,fc,fb,fws)
%============================================================%
%  Continuous Wavelet Transform using Morlet function                
%%%%%%%%%%%%%%%%%%%%%%%%输入%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Sig: 输入信号                                          
%    Scales: 输入的尺度序列 
%    fc: 小波中心频率  （默认为2） 
%    fb: 小波带宽参数   （默认为2）
%    fws: 小波基采样频率  （默认为1） 
%%%%%%%%%%%%%%%%%%%%%%%%输出%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    WT:  morlet小波变换计算结果
%============================================================%
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

if (nargin <= 1)
     error('At least 2 parameter required');
end;
if (nargin ==4)
     fws=1;
elseif (nargin==3) 
     fws=1;
     fb=2;
elseif (nargin==2) 
     fws=1;
     fb=2;  
     fc = 2;
end;

wavsupport=5;                           % 默认morlet小波的支撑区为[-8,8]
nLevel=length(Scales);                  % 尺度的数目
SigLen = length(Sig);                   % 信号的长度
%SigLen=120;
wcoefs = zeros(nLevel, SigLen);         % 分配计算结果的存储单元 

for m = 1:nLevel                        % 计算各尺度上的小波系数  
    a = Scales(m);                                   % 提取尺度参数                              
    t = -round(a*wavsupport):1/fws:round(a*wavsupport);          % 在尺度a的作用区，小波的支撑区会变为[-a*wavsup,a*wavsup]，采样频率为1Hz
    Morl = real(fliplr((pi*fb)^(-1/2)*exp(-i*2*pi*fc*t/a).*exp(-t.^2/(fb*a^2))));     % 计算当前尺度下的小波函数
    temp = conv(Sig,Morl) / sqrt(a);            % 计算信号与当前尺度下小波函数的卷积   
    d=(length(temp)-SigLen)/2;                  % 由于卷积计算所得结果的长度可能远远大于原信号，只需提取按原信号的长度获取提取中间部分的系数
    first = floor(d)+1;                         % 区间的起点
  %  first=floor(length(temp)/3);
    wcoefs(m,:)=temp(first:first+SigLen-1);   
  %  wcoefs(m,:)=temp(1:SigLen);   
end