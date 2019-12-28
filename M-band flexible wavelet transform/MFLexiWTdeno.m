function e=MFLexiWTdeno(xs,x,N,p,q,r1,s1,r2,s2,bet,alp,J,th,sorh)
% xs------noisy signal 染噪信号
% x------original signal 原信号
% N------data length
% p,q,r1,s1,r2,s2 : sampling parameters of wavelet filters
% bet, alp : filter parameters
% J-----number of levels 分解层数
% th----Threshold 阈值，可以是一个序列
% e-----RMS if error 均方根误差
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

nt=length(th);   %阈值序列长度
[F] = CreateFilters2(N,p,q,r1,s1,r2,s2,bet,alp,J);
w=MFLexiWT(xs,p,q,r1,s1,r2,s2,J,F);
for pp=1:nt
    pc=th(pp);
    for j=1:J
        w1{j,1}=wthresh(real(w{j,1}),sorh,pc);
        w1{j,2}=wthresh(real(w{j,2}),sorh,pc);
        w1{j,3}=wthresh(real(w{j,3}),sorh,pc);
        w1{j,4}=wthresh(real(w{j,4}),sorh,pc);
    end
    w1{J+1,1}=w{J+1,1};
    y=iMFLexiWT(xs,w1,N,p,q,r1,s1,r2,s2,F);
    e(pp) = sqrt(mean(mean((y-x).^2)));
end

