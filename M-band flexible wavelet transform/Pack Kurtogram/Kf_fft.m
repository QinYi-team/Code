function K = Kf_fft(x,Nfft,foverlap,opt)%通过短时傅利叶变换法计算每层每段峭度

if rem(log2(Nfft),1)~=0%？这个语句在这里好象不起什么作用
   error('Nfft should contain powers of two only !!')
end
if rem(foverlap,1)~=0 | foverlap ==0%最大频率必须是非零整数？
   error('foverlap must be a non-zero integer !!')
end

N = length(Nfft);%总分层数
L = Nfft(N)*foverlap;%最后一层长度
K = zeros(N,L/2);%初始化峭度矩阵

for i = 1:N
   Window = hanning(Nfft(i));		% bandwidth(3dB) ~ .6/N (small N) --> .7/N (large N)每一层的汉宁窗长度为该层所分段数
   Nw = Nfft(i);%每层窗长
   Noverlap = fix(3*Nw/4); %每层窗重叠长度
   NFFT = 2^nextpow2(Nfft(i)*foverlap);%每层傅利叶变换长度
   temp = Kf_W(x,NFFT,Noverlap,Window,opt);%加窗傅利叶变换计算当前层峭度
   temp = temp(1:NFFT/2);%此时的temp为一列向量
   temp = repmat(temp',L/2/length(temp),1);%
   K(i,:) = reshape(temp,L/2,1)';%将temp中的数变成一列，第i层的峭度
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%