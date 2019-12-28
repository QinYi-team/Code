function [s4,s6] = circshift_segment(s1,l,m,n,N)
%运用循环平移进行分段;s1表示剩余信号，l表示每段长度，m表示循环平移段数，N表示信号长度
%l: the number of points inone period
%m: the number of circulating shifts
%n: the number of points in one circulating shift(it can be set as 1 or 2.) 
%N: data length 
% =========================================================================
%                          Written by Yi Qin
% =========================================================================
s1=s1(1:N);
s3=zeros(N,m);
for i=1:m
    s3(:,i)=circshift(s1,i*n);        %循环平移
end


s4=[];
for i=1:m
    s5=reshape(s3(:,i),l,N/l);      %分段后组成一个矩阵
    s4=[s4,s5];
    s6(:,i)=sum(s5,2)/(N/l);
end