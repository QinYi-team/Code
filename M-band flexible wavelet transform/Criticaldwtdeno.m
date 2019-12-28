function e=Criticaldwtdeno(s,x,wname,n,th,sorh)
% s------染噪信号
% x------原信号
% wname--小波名称
% n-----分解层数
% th----阈值，可以是一个序列
% e-----均方根误差
% =========================================================================
%                          Written by Yi Qin
% =========================================================================
nt=length(th);   %阈值序列长度

% [h,g,rh,rg]=wfilters(wname);
% [r,ww]=freqz(g);
% th=th*std(r);

for pp=1:nt
   thr=th(pp)*ones(1,n);
   y = wdencmp('lvd',s,wname,n,thr,sorh);
   e(pp) = sqrt(mean(mean((y-x).^2)));
end