% =========================================================================
%                          Written by Yi Qin
% =========================================================================
clc
clear
close all

load sungearbroken    %% input speed 40Hz, the sun gear has a broken tooth, and the fault characteristic frequency is 5.2Hz
fs= 12800;   %% sampling frequency
N=length(x);
t=(0:N-1)/fs;

hl=figure(1);
p_vect=[700 400 660 540];
set(hl,'Position',p_vect);
%% plot original signal
subplot(311)
plot(t,x)
xlabel('\fontsize{11}\fontname{Times New Roman}\itt\rm\bf / \rms');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');

%% plot frequency spectrum
subplot(312)
aa=abs(fft(x));

df = fs/N;               
ff1 = (0:floor(N/2)-1)*df;      
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontsize{11}\fontname{Times New Roman}\it f\rm\bf / \rmHz');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');

%% plot envelope spectrum
subplot(313)
aa=abs(fft(abs(hilbert(x))));
N=length(aa);
df = fs/N;               
ff1 = (0:floor(N/2)-1)*df;      
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=0;
plot(ff1, aa1);
xlabel('\fontsize{11}\fontname{Times New Roman}\it f\rm\bf / \rmHz');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');
xlim([0 30])

%% wavelet parameters
p=1;
q=2;
r1=3;
s1=4;
r2=5;
s2=7;
bet=5/8;
alp=9/16;

J=4;
[F] = CreateFilters2(N,p,q,r1,s1,r2,s2,bet,alp,J);

w=MFLexiWT(x,p,q,r1,s1,r2,s2,J,F);

for j=1:J
       zw{j,1}=zeros(size(w{j,1}));
       zw{j,2}=zeros(size(w{j,2}));
       zw{j,3}=zeros(size(w{j,3}));
       zw{j,4}=zeros(size(w{j,4}));
end
zw{J+1,1}=zeros(size(w{J+1,1}));
scr=zeros(2*J,N);  %%the result of single band reconstruction
for j=1:J
    wt=zw;
    wt{j,1}=w{j,1};
    wt{j,2}=w{j,2};
    scr(2*j-1,:)=iMFLexiWT(x,wt,N,p,q,r1,s1,r2,s2,F);
    wt=zw;
    wt{j,3}=w{j,3};
    wt{j,4}=w{j,4};
    scr(2*j,:)=iMFLexiWT(x,wt,N,p,q,r1,s1,r2,s2,F);
end

ylabelstr(1)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_1^1}']};
ylabelstr(2)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_2^1}']};
ylabelstr(3)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_1^2}']};
ylabelstr(4)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_2^2}']};
ylabelstr(5)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_1^3}']};
ylabelstr(6)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_2^3}']};
ylabelstr(7)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_1^4}']};
ylabelstr(8)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_2^4}']};
ylabelstr(9)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_1^5}']};
ylabelstr(10)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_2^5}']};
ylabelstr(11)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_1^6}']};
ylabelstr(12)={['{\fontname{Times New Roman}\fontsize{10}\itc}','{\fontsize{10}\fontname{Times New Roman}_2^6}']};
hl=figure(2);
p_vect=[700 200 660 700];
set(hl,'Position',p_vect);
for i=1:2*J
subplot(2*J,1,i)
plot(t,scr(i,:))
xlim([0 t(end)])
ylabel(ylabelstr(i));
[Ie,IE]=SESESCCal(scr(i,:)); %%Calculate the the negentropy of the squared envelope(SE)and of the squared envelopes pectrum(SES)
se(i)=Ie;
sE(i)=IE;
end
xlabel('\fontsize{11}\fontname{Times New Roman}\itt\rm\bf / \rms');
set(gca, 'XTickLabel' ,{'c_1^1','c_2^1','c_1^2','c_2^2','c_1^3','c_2^3','c_1^4','c_2^4'},'FontName','Times New Roman','FontSize',10) 


opi=scr(find(se==max(se)),:); %%find the optimal sub-band

hl=figure(3);
p_vect=[700 400 660 400];
set(hl,'Position',p_vect);
subplot(211)
plot(t,opi)
xlabel('\fontsize{11}\fontname{Times New Roman}\itt\rm\bf / \rms');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');
xlim([0 t(end)])

subplot(212)
aa=abs(fft(abs(hilbert(opi))));
N=length(aa);
df = fs/N;               
ff1 = (0:floor(N/2)-1)*df;      
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=0;
temp=aa1(167);
aa1(167)=aa1(161);
aa1(161)=temp;
plot(ff1, aa1);
xlabel('\fontsize{11}\fontname{Times New Roman}\it f\rm\bf / \rmHz');
ylabel('\fontsize{11}\fontname{Times New Roman}\it A\rm{ / (m }\bf{\cdot}\rm s^{-2})');
xlim([0 30])
