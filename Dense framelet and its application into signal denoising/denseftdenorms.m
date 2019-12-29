function e=denseftdenorms(s,x,af,n,th,sorh)
% s------noisy signal 染噪信号
% x------original signal 原信号
% af-----analysis filter bank 分析滤波器组
% n-----the number of levels 分解层数
% th----threshold 阈值，可以是一个序列
% sorh----'s' soft thresholding or 'h' hard thresholding 软阈值或硬阈值
% e-----root-mean-square(RMS error)均方根误差
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

sf = af(end:-1:1,:);
nt=length(th);   %阈值序列长度
h1=af(:,2);
h2=af(:,3);
[r1,ww]=freqz(h1);
[r2,ww]=freqz(h2);

w = denseft(s, n, af);

for pp=1:nt
    w1=w;
    thr=th(pp)*ones(1,2*n);
    for k = 1:n
        p   = 2*k-1;
        pc  = thr(p);        % thresholds
        cfs = w1{k}{1};
        cfs = wthresh(cfs,sorh,std(r1)*pc);
        w1{k}{1}= cfs;
        p   = 2*k;
        pc  = thr(p);        % thresholds
        cfs = w1{k}{2};
        cfs = wthresh(cfs,sorh,std(r2)*pc);
        w1{k}{2}= cfs;
    end
    y = idenseft(w1, n, sf);
    e(pp) = sqrt(mean(mean((y-x).^2)));
end


