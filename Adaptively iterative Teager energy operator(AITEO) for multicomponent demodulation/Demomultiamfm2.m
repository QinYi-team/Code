% =========================================================================
%                          Written by Yi Qin
% =========================================================================
clc;
clear all;
close all;

%% 原始信号
% fs=1600;
fs=1500;
t=0:1/fs:3-1/fs;
k=0;   %the amount of component
f0=20;
f1=500;
f2=600;
a1=1+0.3*cos(2*pi*f0*t);
b1=sin(2*pi*140*t+2*pi*40*t.^2); %%chirp
a2=0.3+0.1*cos(1*pi*f0*t);
b2=sin(2*pi*180*t+2*pi*70*t.^2); %%chirp

%%%%%%%%%正弦调制
x1=1*a1.*b1;
x2=a2.*b2;
x=x1+1*x2+0*randn(1,length(x1));
s=x;
hl=figure(1);
p_vect=[700 400 660 240];
set(hl,'Position',p_vect);
plot(t,x,'-b')
ylabel('A')
xlabel('t/s')
xlim([0 t(end)])

oenergy=sum(abs(hilbert(x)).^2)/length(x);
oen=oenergy;
%% 初始化k
k=0;
%初始化信号延拓数
Nl_ex = 450;  %%在450-520之间变动较好，450效果最好
Nr_ex = Nl_ex;
%%延拓170
% 原始信号延拓
x = ex_sig(x,Nl_ex,Nr_ex);


%% 循环条件
while ~iht_stop(k)
    %加窗
    %     x = window(x);
    if k==2
        [env,w]=teodemodu(x);
    elseif k==1
        [env,w]=teodemodu(x);
    else
        [env,w]=teodemodu(x);  %滤波器参数0.4，0.5最好，可进行滤波teodemodu2，也可不用
    end
    
    
    % % %     高通滤波
    [b,a,bs,as,isbs] =  arg_filter(env,k);
    % a=1;
    if isbs
        en=filtfilt(b,a,env);  %低通滤波
        
    else
        en=filtfilt(b,a,env); %低通滤波
    end
    
    res=env-en;
    
    Nl_re =80;   %%%60时效果最好
    Nr_re = Nl_re;
    res = re_ex_sig(res,Nl_re,Nr_re);
    
    
    %画图前去除延拓
    Nx = length(res);
    %计算信号起止点
    Nl_ex = Nl_ex - Nl_re;
    Nr_ex = Nr_ex - Nr_re;
    l_idx = Nl_ex+1;
    r_idx = Nx - Nr_ex;
    
    %保存相位和幅值
    k = k+1;
    am(k,:) = en(1,l_idx:r_idx);
    insf(k,:)=w(1,l_idx:r_idx)*fs;
    %     phaze(k,:) = ph(1,l_idx:r_idx);
    
    %画图
    figure(k+10)
    plot(t,am(k,:));
    ylabel('幅值 A')
    xlabel('时间 t/s')
    
    
    
    
    %  amplitude spectrum of envelop component
    a=abs(fft(env(1,l_idx:r_idx)));
    N=length(env(1,l_idx:r_idx));
    df = fs/N;               %frequency resolution Hz
    f1 = (0:floor(N/2)-1)*df;
    a1= 2*a(1:floor(N/2))/N;
    a1(1)=a1(1)/2;          %frequency at zero won't double
    a1(1) = 0;
    %  amplitude spectrum of residue component
    Yres=abs(fft(res(1,l_idx:r_idx)));
    Nres = length(res(1,l_idx:r_idx));
    dfres = fs/Nres;
    fres= (0:floor(Nres/2)-1)*dfres;
    yres = Yres(1:floor(Nres/2))*2/Nres;
    yres(1) = 0;
    %  amplitude spectrum of first decomposed signal
    Yam=abs(fft(am(k,:)));
    Nam = length(am(k,:));
    dfam = fs/Nam;
    fam= (0:floor(Nam/2)-1)*dfam;
    yam = Yam(1:floor(Nam/2))*2/Nam;
    yam(1) =yam(1)/2;
    
    figure(k+20)
    plot(fam, yam);
    ylabel('幅值 A')
    xlabel('频率 f/Hz')
    xlim([0,200]);
    
    % 迭代
    x= res;
    oen1=oen;
    oen=sum(abs(am(k,:)).^2)/length(am(k,:));
    ratio(k)=oen/oenergy;
    ratio1(k)=oen/oen1;
end
if1=insf(1,:);
[b,a,bs,as,isbs] =  arg_filter1(if1);
if isbs
    if1=filtfilt(b,a,if1);  %低通滤波
    if1=filtfilt(bs,as,if1); %带阻滤波
else
    if1=filtfilt(b,a,if1); %低通滤波
end


if21=insf(2,:)-if1;
[b,a,bs,as,isbs] =  arg_filter1(if21);
if isbs
    if21=filtfilt(b,a,if21);  %低通滤波
    if21=filtfilt(bs,as,if21); %带阻滤波
else
    if21=filtfilt(b,a,if21); %低通滤波
end



if22=insf(2,:)+if1;
[b,a,bs,as,isbs] =  arg_filter1(if22);
if isbs
    if22=filtfilt(b,a,if22);  %低通滤波
    if22=filtfilt(bs,as,if22); %带阻滤波
else
    if22=filtfilt(b,a,if22); %低通滤波
end


figure(31)
fr=if1;
aa=abs(fft(fr));
N=length(fr);
df = fs/N;                      %频域分辨率 Hz
ff1 = (0:floor(N/2)-1)*df;      %频域序列
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;                %零频分量不乘2
plot(ff1, aa1);
title('the spectrum of 1 component IF');
xlim([0,200]);

figure(32)
fr=if21;
aa=abs(fft(fr));
N=length(fr);
df = fs/N;                      %频域分辨率 Hz
ff1 = (0:floor(N/2)-1)*df;      %频域序列
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;                %零频分量不乘2
subplot(2,1,1)
plot(ff1, aa1);
title('the spectrum of 1 of 2 component IF');
xlim([0,200]);
fr=if22;
aa=abs(fft(fr));
N=length(fr);
df = fs/N;                      %频域分辨率 Hz
ff1 = (0:floor(N/2)-1)*df;      %频域序列
aa1= 2*aa(1:floor(N/2))/N;
aa1(1)=aa1(1)/2;                %零频分量不乘2
subplot(2,1,2)
plot(ff1, aa1);
title('the spectrum of 2 of 2 component IF');
xlim([0,200]);

figure(34);
subplot(211)
plot(t,if1)
ylim([0 800])
subplot(212)
plot(t,if22)
ylim([0 800])

hl=figure(7);
p_vect=[440 200 560 210];
set(hl,'Position',p_vect);
plot(t,am(1,:),'-r');
hold on;
plot(t,am(2,:),'--');
hold off;
haaxes1=gca;
haaxes=get(hl,'CurrentAxes');

xlabel('\fontname{Times New Roman}\it t\rm\bf / \rms');
ylabel('\fontname{Times New Roman}Amplitude');
legend('\fontname{Times New Roman}Instantaneous amplitude of \it z\rm_1(\itt\rm)','\fontname{Times New Roman}Instantaneous amplitude of \it z\rm_2(\itt\rm)','Location','NorthWest')
ylim([0 2])

hl=figure(8);
p_vect=[440 200 560 210];
set(hl,'Position',p_vect);
plot(t,if1,'-r');
hold on;
plot(t,if22,'--');
hold off;
haaxes1=gca;
haaxes=get(hl,'CurrentAxes');
xlabel('\fontname{Times New Roman}\it t\rm\bf / \rms');
ylabel('\fontname{Times New Roman}Frequency  Hz');
legend('\fontname{Times New Roman}Instantaneous frequency of \it z\rm_1(\itt\rm)','\fontname{Times New Roman}Instantaneous frequency of \it z\rm_2(\itt\rm)','Location','NorthWest')


%% 信号or包络延拓
function x = ex_sig(x,Nl,Nr)
if Nl == 0
    x =x;
else
    
    Nx = length(x);
    for il = 1:Nl
        xl(il) = x(Nl+2-il);
    end
    for ir = 1:Nr
        xr(ir) = x(Nx-ir);
    end
    x = [xl,x,xr];
end
end
%% 去除信号or包络延拓
function x = re_ex_sig(x,Nl,Nr)
Nx = length(x);
x = x(1,Nl+1:Nx-Nr);
end

%% 加窗

function x = window(x)
n = length(x);
size(x)
w = hamming(n);
size(w)
x = x.*w';
end
%% 停止条件
function b_s = iht_stop(k)
if k >= 2
    b_s = 1;
else b_s = 0;
end
end

%% 滤波器设计
function [b,a,bs,as,isbs] = arg_filter(en,k)
% fft of en.
fs = 1500;
a=abs(fft(en));
N=length(en);
df = fs/N;               %frequency resolution Hz
f1 = (0:floor(N/2)-1)*df;
a1= 2*a(1:floor(N/2))/N;
a1(1)=a1(1)/2;


%%%%%%
figure(100)
plot(f1,a1);
%%%%%%
Nf = floor(N/20);
maxind=[];
for i = 2: Nf - 1
    if 1*a1(i - 1) < a1(i) && a1(i) > 1*a1(i + 1)  %%去除一些震荡极值点
        maxind=[maxind,i];
    end
end

% 找寻第一个峰值

maxVec = find(a1==(max(a1(maxind))));
%     maxVec = find(a1==(max(a1(2:Nf))));

fce=maxVec/N*2;

if maxVec==maxind(1)
    isbs=0;
    bs=0;
    as=0;
    
    fce0=fce+0.002;
    fce1=fce+0.007;
    f=[fce0 fce1];
    a=[1 0];
    rp=0.5;
    rs=30;
    [n,wn]=buttord(fce0,fce1,rp,rs);
    [b,a]=butter(n,wn);
    while sum(abs(b))/length(b)<1e-10
        rs=rs-2;
        [n,wn]=buttord(fce0,fce1,rp,rs);
        [b,a]=butter(n,wn);
    end
else
    isbs=1;
    bs=0;
    as=0;
    fce0=fce+0.003;
    
    fce1=fce+0.008;
    f=[fce0 fce1];
    a=[1 0];
    rp=1;
    rs=50;
    drp=10^((rp/20)-1)/(10^(rp/20)+1);
    drs=10^(-rs/20);
    dev=[drp drs];
    
    [n,fo,ao,w]=remezord(f,a,dev);
    %     ao(1:2)=1.033;
    b=remez(n,fo,ao,w);
    a=1;
    if k==1
        bs=0;
        as=0;
        fce0=fce+0.002;
        fce1=fce+0.007;
        f=[fce0 fce1];
        a=[1 0];
        rp=1;
        rs=30;
        [n,wn]=buttord(fce0,fce1,rp,rs);
        [b,a]=butter(n,wn);
        while sum(abs(b))/length(b)<1e-10
            rs=rs-2;
            [n,wn]=buttord(fce0,fce1,rp,rs);
            [b,a]=butter(n,wn);
        end
    end
end
end

function [b,a,bs,as,isbs] = arg_filter1(fr)
% fft of en.
fs = 1500;
a=abs(fft(fr));
N=length(fr);
df = fs/N;               %frequency resolution Hz
f1 = (0:floor(N/2)-1)*df;
a1= 2*a(1:floor(N/2))/N;
a1(1)=a1(1)/2;


%%%%%%
figure(200)
plot(f1,a1);
%%%%%%
Nf = floor(N/20);
maxind=[];
for i = 2: Nf - 1
    if 1.0*a1(i - 1) < a1(i) && a1(i) > 1.0*a1(i + 1)  %%去除一些震荡极值点
        maxind=[maxind,i];
    end
end

% 找寻第一个峰值

maxVec = find(a1==(max(a1(maxind))));

fce=maxVec/N*2;

if maxVec==maxind(1)
    isbs=0;
    bs=0;
    as=0;
    
    fce0=fce+0.001;
    fce1=fce+0.003;
    f=[fce0 fce1];
    a=[1 0];
    rp=1;
    rs=20;
    [n,wn]=buttord(fce0,fce1,rp,rs);
    [b,a]=butter(n,wn);
    while sum(abs(b))/length(b)<1e-10
        rs=rs-1;
        [n,wn]=buttord(fce0,fce1,rp,rs);
        [b,a]=butter(n,wn);
    end
else
    %%%%%%%%低通滤波器设计%%%%%%%%%
    isbs=1;
    fce0=fce+0.002;
    fce1=fce+0.007;
    f=[fce0 fce1];
    a=[1 0];
    rp=1;
    rs=20;
    [n,wn]=buttord(fce0,fce1,rp,rs);
    [b,a]=butter(n,wn);
    while sum(abs(b))/length(b)<1e-10
        rs=rs-1;
        [n,wn]=buttord(fce0,fce1,rp,rs);
        [b,a]=butter(n,wn);
    end
    %%%%%%%%带阻滤波器设计%%%%%%%%%
    fce0=2/fs;
    fce1=fce-2/fs;
    fce2=8/fs;
    fce3=fce-8/fs;
    wp=[fce0 fce1];
    ws=[fce2 fce3];
    rp=0.5;
    rs=20;
    [n,wn]=buttord(wp,ws,rp,rs);
    [bs,as]=butter(n,wn,'stop');
    while sum(abs(bs))/length(bs)>1e1 || sum(abs(as))/length(as)>1e1
        rs=rs-1;
        [n,wn]=buttord(wp,ws,rp,rs);
        if n==1
            n=2;
            wl=(fce0+ fce2)/2;
            wh=(fce1+ fce3)/2;
            [bs,as]=butter(n,[wl wh],'stop');
            while sum(abs(bs))/length(bs)>1e1 && n<7
                n=n+1;
                [bs,as]=butter(n,[wl wh],'stop');
            end
        else
            [bs,as]=butter(n,wn,'stop');
        end
    end
end
end

function [b,a,bs,as,isbs] = arg_filter2(fr)
% fft of en.
fs = 1500;
a=abs(fft(fr));
N=length(fr);
df = fs/N;               %frequency resolution Hz
f1 = (0:floor(N/2)-1)*df;
a1= 2*a(1:floor(N/2))/N;
a1(1)=a1(1)/2;


%%%%%%
figure(200)
plot(f1,a1);
%%%%%%
Nf = floor(N/20);
maxind=[];
for i = 2: Nf - 1
    if 1.1*a1(i - 1) < a1(i) && a1(i) > 1.1*a1(i + 1)  %%去除一些震荡极值点
        maxind=[maxind,i];
    end
end

% 找寻第一个峰值

maxVec = find(a1==(max(a1(maxind))));

fce=maxVec/N*2;

if maxVec==maxind(1)
    isbs=0;
    bs=0;
    as=0;
    
    fce0=fce+0.002;
    fce1=fce+0.007;
    f=[fce0 fce1];
    a=[1 0];
    rp=1;
    rs=20;
    [n,wn]=buttord(fce0,fce1,rp,rs);
    [b,a]=butter(n,wn);
    while sum(abs(b))/length(b)<1e-10
        rs=rs-1;
        [n,wn]=buttord(fce0,fce1,rp,rs);
        [b,a]=butter(n,wn);
    end
else
    %%%%%%%%低通滤波器设计%%%%%%%%%
    isbs=1;
    fce0=fce+0.003;
    fce1=fce+0.008;
    f=[fce0 fce1];
    a=[1 0];
    rp=2;
    rs=50;
    drp=10^((rp/20)-1)/(10^(rp/20)+1);
    drs=10^(-rs/20);
    dev=[drp drs];
    [n,fo,ao,w]=remezord(f,a,dev);
    ao(1)=1;
    b=remez(n,fo,ao,w);
    a=1;
    %%%%%%%%带阻滤波器设计%%%%%%%%%
    fce0=2/fs;
    fce1=fce-2/fs;
    fce2=8/fs;
    fce3=fce-8/fs;
    wp=[fce0 fce1];
    ws=[fce2 fce3];
    rp=2;
    rs=20;
    [n,wn]=buttord(wp,ws,rp,rs);
    [bs,as]=butter(n,wn,'stop');
    while sum(abs(bs))/length(bs)>1e1 || sum(abs(as))/length(as)>1e1
        rs=rs-1;
        [n,wn]=buttord(wp,ws,rp,rs);
        if n==1
            n=2;
            wl=(fce0+ fce2)/2;
            wh=(fce1+ fce3)/2;
            [bs,as]=butter(n,[wl wh],'stop');
            while sum(abs(bs))/length(bs)>1e1 && n<7
                n=n+1;
                [bs,as]=butter(n,[wl wh],'stop');
            end
        else
            [bs,as]=butter(n,wn,'stop');
        end
    end
end
end