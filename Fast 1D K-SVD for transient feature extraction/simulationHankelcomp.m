% =========================================================================
%                          Written by Yi Qin
% =========================================================================
tic;
clc;
clear;
close all;


%% 参数设置

%信号初始参数
fs = 2000;             %采样频率
N = 2000;              %采样点数
T = 1/fs;               %采样时间
t = (0:N-1)*T;          %时间序列
%自适应OMP参数
a=2;                    %过完备字典参数
c=5;                    %自适应稀疏度判据
%Hankel矩阵分段参数
l=500;                     %每段信号长度
m=100;                      %Hankel矩阵行数
n=401;                      %Hankel矩阵列数
%K-SVD降噪参数(参照K-SVD函数)
param.K=300;                %原子个数
param.numIteration = 10;    %迭代次数
param.errorFlag = 1;        
param.errorGoal =0.3;       %errorGoal可调
param.preserveDCAtom =0;
param.displayProgress = 1;
param.InitializationMethod ='DataElements';

%% 生成仿真信号
s=simulation_signal(N,t);

%% 运用OMP去除谐波和调制分量  
D1=FourierDict(N,a);    %定义傅里叶字典
s1 = adapomp(D1,s,c);   %自适应OMP
plot1(t,s1,21,1);         %画出去除谐波和调制分量后时域图  

%% Hankel矩阵分段
S = Hankel_matrix(s1,l,m,n,N);

%% K-SVD降噪
for i=1:N/l
    [D{i},output]=KSVD(S(:,:,i),param);
    A2{i} = output.CoefMatrix;          %这里CoefMatrix是稀疏矩阵，所以不能够用N维索引
    X2 = D{i} * A2{i};                  %outputCoefMatrix需要保存出来以备做阈值处理
    x2((i-1)*l+1:i*l) = ReHankel(X2);   %恢复信号
end

%% 画图
plot2(t,x2,fs,22,300,1);              
toc;