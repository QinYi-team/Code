% =========================================================================
%                          Written by Yi Qin
% =========================================================================
tic;
clc;
clear;
close all;


%%parameter setting 参数设置

%Parameters of signal 信号初始参数
fs = 25600;             %采样频率
N = 16284;              %采样点数
T = 1/fs;               %采样时间
t = (0:N-1)*T;          %时间序列
%Parameters of adaptive OMP自适应OMP参数
a=2;                    %Parameter of overcomplete dictionary 过完备字典参数
c=5;                    %Parameter for stopping criterion 自适应稀疏度判据
%分段参数
l=236;                  %the number of points inone period 每段的长度（按周期分段）
m=10;                   %the number of circulating shifts 循环平移需要的段数
n=2;                    %the number of points in one circulating shift(it can be set as 1 or 2.) 每次平移点数
%Parameters of adaptive K-SVD 自适应K-SVD参数(参照K-SVD函数)
param.K=610;
param.numIteration = 10;
param.errorFlag = 1;
param.errorGoal =1.16;                      %KSVD的参数，重点是errorGoal
param.preserveDCAtom =0;
param.displayProgress = 1;
param.InitializationMethod ='GivenMatrix';
k=600;                                      %the number of atoms 原信号中取出作为初始字典的原子个数

%% faulty bearing signal加载信号
load bearingout1800;
ss=s(1:N);
ss=ss';
plot1(t,ss,1,0.6);         %%画出原信号时域图 

%% Remove harmonic components by OMP 运用OMP去除谐波和调制分量  
D1=FourierDict(N,a);    %定义傅里叶字典
s1 = adapomp(D1,ss,c);   %自适应OMP

%% period segmentation and circulating shift 运用循环平移进行分段
[s4,s6]=circshift_segment(s1,l,m,n,N);

%%1D K-SVD with adaptive transient dictionary自适应K-SVD算法
d=randperm(size(s4,2));
y1=s4(:,d(1:k));
y=[y1,s6];
param.initialDictionary = y;   %构造初始字典

[D,output]=KSVD(s4,param);
X = D * output.CoefMatrix;             %KSVD后信号矩阵

%% Signal recovery分段信号还原
x=zeros(N,m);
x=reshape(X,N,m);

%% Perform hard thresholding on dictionary 对字典进行阈值处理
 [A]=hard(D,0.2);              %对字典做硬阈值处理
 X2=A * output.CoefMatrix;
 x2=zeros(N,m);
 x2=reshape(X2,N,m);
 
 %% 画图
 plot2(t,x2(:,1),fs,2,600,0.6);   
 toc;