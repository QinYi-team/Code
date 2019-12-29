% =========================================================================
%                          Written by Yi Qin
% =========================================================================
clc;
clear;
close all;

load hs13; %%Framelets constructed by Qin in 'Higher density wavelet frames with symmetric low-pass and band-pass filters'
af2=af;

load spline2   %%Proposed spline framelets with two vanishing moments
af3=af;

load spline3   %%Proposed spline framelets with three vanishing moments
af4=af;
% af=[h0' h1' h2'];
load HDDWT43 %%Framelets constructed by I.W. Selesnick in 'A higher density discrete wavelet transform'
wname='db3'; %%Daubechies wavelet with three vanishing moments

nnum=100;
n=5; %decomposition level
load Pieceregular
figure(1)
plot(x)
sorh='h';
nstd=0.145;
%%%With the increase noise level, threshold should be changed
if nstd==0.1
    if n==5
        th=0.2:0.025:0.6;
    else
        th=0.2:0.025:0.6;
    end
elseif nstd==0.2
    if n==5
        th=0.5:0.05:1;
    else
        th=0.5:0.025:0.9;
    end
elseif nstd==0.3
    if n==5
        th=0.8:0.05:1.4;
    else
        th=0.8:0.05:1.4;
    end
elseif nstd==0.15
    if n==5
        th=0.35:0.05:0.85;
    else
        th=0.35:0.05:0.85;
    end
else
    th=0.35:0.05:0.85;
end

osnr=0;
for i=1:nnum
    s=x+nstd*randn(1,length(x));
    noistd=std(s-x);  %Noise variance
    osnr=osnr+SNR(x,s);    %Calculate the accumulation of SNR
    
    ecdwta(i,:)=Criticaldwtdeno(s,x,wname,n,th,sorh);
    ehda(i,:)=denseftdenorms(s,x,af,n,th,sorh);
    ehdda2(i,:)=denseftdenorms(s,x,af2,n,th,sorh);
    ehdda3(i,:)=denseftdenorms(s,x,af3,n,th,sorh);
    ehdda4(i,:)=denseftdenorms(s,x,af4,n,th,sorh);
end

ecdwt=sum(ecdwta)/nnum;
ehd=sum(ehda)/nnum;
ehdd2=sum(ehdda2)/nnum;
ehdd3=sum(ehdda3)/nnum;
ehdd4=sum(ehdda4)/nnum;

asnbr=osnr/nnum;
hf_fig=figure(2);
p_vect=[600 300 621 505];
set(hf_fig,'Position',p_vect);
plot(th,ecdwt,'-k*','MarkerSize',6)
hold on;
plot(th,ehd,'-gd','MarkerSize',6)
plot(th,ehdd2,'-ro','MarkerSize',6)
plot(th,ehdd3,'-mv','MarkerSize',6)
plot(th,ehdd4,'-bs','MarkerSize',6)
hold off;

xlim([min(th),max(th)])
ymin=min([ehd,ecdwt,ehdd2,ehdd3,ehdd4]);
ymax=max([ehd,ecdwt,ehdd2,ehdd3,ehdd4]);
off=(ymax-ymin)/10;
ylim([ymin-off,ymax+off])
set(gca,'FontName','Times New Roman','FontSize',14);
xlabel('\fontsize{14}\fontname{Times New Roman}THRESHOLD');
ylabel('\fontsize{14}\fontname{Times New Roman}RMS ERROR');
legend('\fontname{Times New Roman}\fontsize{12}Daubechies wavelets','\fontname{Times New Roman}\fontsize{12}Framelets constructed by I.W. Selesnick','\fontname{Times New Roman}\fontsize{12}Framelets constructed by Qin',...
    '\fontname{Times New Roman}\fontsize{12}Spline Framelets in Example 1','\fontname{Times New Roman}\fontsize{12}Spline Framelets in Example 2','Location','NorthWest')
