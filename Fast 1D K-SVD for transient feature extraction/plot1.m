function [A] = plot1(t,s,l,k)
% =========================================================================
%                          Written by Yi Qin
% =========================================================================
%画单张图片；s表示信号，t表示时间序列，l表示图号
hl=figure(l);         %画出分离谐波和调制分量信号图
p_vect=[700 400 660 200];
set(hl,'Position',p_vect);
plot(t,s,'linewidth',2);
set(gca,'fontsize',15)
xlabel('\fontsize{15}\fontname{Times New Roman}\itt\rm\bf / \rms');
ylabel('\fontsize{15}\fontname{Times New Roman}\it A');
xlim([0 k]);
