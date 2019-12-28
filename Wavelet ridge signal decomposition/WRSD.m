
function D=WRSD(y,a0,n,fb,fc,fs,lf,num)
%%y-----original signal 原始信号
%%a0-----initial scale 初始尺度
%%n----- extended length 信号延拓长度
%%lf----Low pass cutoff frequency of IF 瞬时频率低通滤波截止频率
%%%%fb,fc--------------------Parameters of Morlet wavelet Morlet小波时间带宽参数和中心频率
%%fs---sampling frequency 采样频率
%%num---the  number of target components 分解分量个数
%%D-----decomposition result 分解结果
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

thr=0.005;
MAXITERATIONS=10;

for i=1:num
ey=symextend(y,n); %%信号延拓
[wra,as]=mywaveridge2(ey,fb,fc,thr,a0,MAXITERATIONS);
f1=fs./wra;
F=f1'/fs;
F=lpf(F,0.02);
[xi,Ai,phi]=synchdem(ey',2*pi*F,0.02);
x=deextend(xi,n);
y=y-x;
a0=a0*1.2;
D(i,:)=x;
end



function ex=symextend(x,n)
x=x(:);
N=length(x);
x1=fliplr(x(1:n));
x2=fliplr(x(N-n+1:N));
ex=[x1;x;x2];
ex=ex';

function x=deextend(ex,n)
ex=ex(:);
N=length(ex);
x=ex(n+1:N-n);
x=x';