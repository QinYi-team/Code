function [Ie,IE]=SESESCCal(x)
% =========================================================================
%                          Written by Yi Qin
% =========================================================================
x=abs(x);
cev=hilbert(x);
aev=(abs(cev)).^2;
aevs=abs(fft(aev));
t1=aev.^2./mean(aev.^2);
t2=t1(find(t1~=0));
Ie=mean(t2.*log(t2));
% Ie=mean(aev.^2./mean(aev.^2).*log(aev.^2./mean(aev.^2)));  %% ±”ÚÏÿ
t3=aevs.^2./mean(aevs.^2);
t4=t1(find(t3~=0));
IE=mean(t4.*log(t4));
% IE=mean((aevs.^2./mean(aevs.^2)).*log(aevs.^2./mean(aevs.^2))); %%∆µ”ÚÏÿ