function [A]=OMP(D,X,L)
%=============================================
% Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: 
%       D - the dictionary (its columns MUST be normalized).
%       X - the signals to represent
%       L - the max. number of coefficients for each signal.
% output arguments: 
%       A - sparse coefficient matrix.
%=============================================
[n,P]=size(X);
[n,K]=size(D);
for k=1:1:P
    a=[];
    x=X(:,k);               %分别计算信号的每一列
    residual=x;             %初始将x定义为残差
    indx=zeros(L,1);        %定义索引向量，因为稀疏度为L，所以列数为L
    for j=1:1:L
        proj=D'*residual;   %用字典转置乘以残差（相当于用每个字典原子与信号求内积）
        [maxVal,pos]=max(abs(proj));    %提取出其中内积最大的原子的列号
        pos=pos(1);                     %若有多个内积相同，取出第一个赋值给pos
        indx(j)=pos;                    %将pos的值赋予j次迭代的索引中
        a=pinv(D(:,indx(1:j)))*x;       %对提取出来的原子求伪逆，并与信号相乘得到对应的系数
        residual=x-D(:,indx(1:j))*a;    %
        if sum(residual.^2) < 1e-6
            break;
        end
    end
    temp=zeros(K,1);
    temp(indx(1:j))=a;
    A(:,k)=sparse(temp);
end
return;
