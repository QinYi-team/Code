
function x=impulseextraction(y,totalscal)
%% y---original signal原信号
%% totalscal---the number of scales总尺度数目
%% x---extracted repetitive transients提取的冲击信号
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

Scales= linspace(1,40,totalscal);
%%%Parameter optimization, but this process is time-consuming 参数优化,比较费时
trop=zeros(1,78);
for i=1:78
    vfc(i)=0.2+i*0.1;
    wcoefs = myrmorletcwt(y,Scales,vfc(i),50,1);
    for k=1:length(Scales)
        energy(k)=sum(abs(wcoefs(k,:)).^1);
    end
    tenergy=sum(energy);
    for k=1:length(Scales)
        p(k)=energy(k)/tenergy;
        if p(k)==0
            trop(i)=trop(i)+0;
        else
            trop(i)=trop(i)-2*p(k)*log(p(k));
        end
    end
end

ofc=vfc(find(trop==min(trop)))  %% optimized central frequency
trop=zeros(1,200);
for j=1:200
    vfb(j)=0.2*(j+5);
    wcoefs = myrmorletcwt(y,Scales,ofc,vfb(j),1);
    for k=1:length(Scales)
        energy(k)=sum(abs(wcoefs(k,:)).^1);
    end
    tenergy=sum(energy);
    for k=1:length(Scales)
        p(k)=energy(k)/tenergy;
        if p(k)==0
            trop(j)=trop(j)+0;
        else
            trop(j)=trop(j)-2*p(k)*log(p(k));
        end
    end
end
ofb=vfb(find(trop==min(trop)))  %% optimized bandwidth parameter

wcoefs = myrmorletcwt(y,Scales,ofc,ofb,1);

%%Calculate the reconstruction coefficients计算重构系数
rsig=0;
for i=1:length(Scales)
   rsig=rsig+wcoefs(i,:)*Scales(i)^-1.5;
end
cf=std(y)/std(rsig);


for i=1:length(Scales)
     coef=wcoefs(i,:)*Scales(i)^(0.5);
    morkur(i)=kurtosis(coef);
end

wcoefs1=zeros(size(wcoefs));
x=0;
mmkur=max(morkur);

%%Choose characteristic scale and reconstruct signal 特征尺度选择与信号重构
for i=1:length(Scales)
    if morkur(i)>0.75*mmkur
        coef=wcoefs(i,:);
        dcoef=wthresh(coef,'h',2.3*std(coef));
        x=x+cf*dcoef*Scales(i)^-1.5;
    end
end