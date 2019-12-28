% =========================================================================
%                          Written by Yi Qin
% =========================================================================
close all
clear
clc
%%%Comparison with Biorthogonal wavelet transform and Orthogonal wavelet transform
wname='bior3.9';         %Biorthogonal wavelet
wname1='db2';         %Orthogonal wavelet

%%the parameters of M-band flexible wavelet transfrom in Example 1
pp=2;
qq=3;
rr1=7;
ss1=9;
rr2=rr1;
ss2=ss1;
bet1=8/9;
alp1=13/18;

%%the parameters of M-band flexible wavelet transfrom in Example 2
p=1;
q=2;
r1=3;
s1=4;
r2=5;
s2=7;
bet=5/8;
alp=9/16;

J=6; %the number of levels

load PieceRegular
N=length(x);
sorh='h';
nstd=0.28;
th=0.4:0.05:1.1;
nnum=100;
osnr=0;
for i=1:nnum
    xs=x+nstd*randn(1,N);
    osnr=osnr+SNR(x,xs);    %–≈‘Î±»
    ecdwta(i,:)=Criticaldwtdeno(xs,x,wname,J,th,sorh);
    ecdwta1(i,:)=Criticaldwtdeno(xs,x,wname1,J,th,sorh);
    eRMAnDa(i,:)=MFLexiWTdeno(xs,x,N,pp,qq,rr1,ss1,rr2,ss2,bet1,alp1,J,th,sorh);  %%M-band flexible wavelet transfrom in Example 1
    eRMAnDa1(i,:)=MFLexiWTdeno(xs,x,N,p,q,r1,s1,r2,s2,bet,alp,J,th,sorh);  %%M-band flexible wavelet transfrom in Example 2
end
asnbr=osnr/nnum;
ecdwt=sum(ecdwta)/nnum;
ecdwt1=sum(ecdwta1)/nnum;
eRMAnD=sum(eRMAnDa)/nnum;
eRMAnD1=sum(eRMAnDa1)/nnum;

hf_fig=figure(1);
p_vect=[100 20 621 505];
set(hf_fig,'Position',p_vect);            
plot(th,ecdwt,'-g*','MarkerSize',6)
hold on;
plot(th,ecdwt1,'-cd','MarkerSize',6)
plot(th,eRMAnD,'-ro','MarkerSize',6)
plot(th,eRMAnD1,'-kv','MarkerSize',6)
hold off;
set(gca,'FontName','Times New Roman','FontSize',11);
xlabel('\fontsize{11}\fontname{Times New Roman}THRESHOLD');
ylabel('\fontsize{11}\fontname{Times New Roman}RMS ERROR');
legend('Biorthogonal wavelet transform','Orthogonal wavelet transform','M-band flexible wavelet transfrom in Example 1','M-band flexible wavelet transfrom in Example 2')


