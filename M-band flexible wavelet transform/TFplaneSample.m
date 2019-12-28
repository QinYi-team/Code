% =========================================================================
%                          Written by Yi Qin
% =========================================================================
clc;
clear;
close all
%%  M-band flexible wavelet transform samples the time-frequency plane
p=5;
q=9;
r1=2;
s1=3;
r2=r1;
s2=s1;
bet=5/9;
alp=11/18;
J=4;
dto=1;
dt(1)=s1/r1*dto;
bcf(1)=((1-bet)*pi+r1/s1*pi)/2;
hcf(1)=(alp*pi+pi)/2;
t=0:dt(1):20;
yb=bcf(1)*ones(1,length(t));
yh=hcf(1)*ones(1,length(t));
hl=figure(1)
p_vect=[540 400 560 300];
set(hl,'Position',p_vect);
plot(t,yh,'v','MarkerEdgeColor','k',...
    'MarkerFaceColor','k',...
    'MarkerSize',4) 
hold on;   
plot(t,yb,'o','MarkerEdgeColor','k',...
    'MarkerFaceColor','k',...
    'MarkerSize',4)

        
for i=1:J-1
    dto=q/p*dto;
    dt(i+1)=s1/r1*dto;
    bcf(i+1)=((p/q)^i*(1-bet)*pi+(p/q)^i*r1/s1*pi)/2;
    hcf(i+1)=((p/q)^i*alp*pi+(p/q)^i*pi)/2;
    t=0:dt(i+1):20;
    yb=bcf(i+1)*ones(1,length(t));
    yh=hcf(i+1)*ones(1,length(t));
    plot(t,yh,'v','MarkerEdgeColor','k',...
    'MarkerFaceColor','k',...
    'MarkerSize',4)    
    plot(t,yb,'o','MarkerEdgeColor','k',...
    'MarkerFaceColor','k',...
    'MarkerSize',4)
end
hold off
set(gca,'xticklabel',['1']);
set(gca,'yticklabel',[]);
 xlim([-1 20]) 
ylim([0 3]) 
set(gca,'Box','on');
xlabel('\fontname{Times New Roman}Time');
ylabel('\fontname{Times New Roman}Frequency');
