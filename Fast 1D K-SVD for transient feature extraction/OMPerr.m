function [A]=OMPerr(D,X,errorGoal); 
%=============================================
% Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: D - the dictionary
%                  X - the signals to represent
%                  errorGoal - the maximal allowed representation error for
%                  each siganl.
% output arguments: A - sparse coefficient matrix.
%=============================================
[n,P]=size(X);
[n,K]=size(D);
E2 = errorGoal^2*n;
maxNumCoef = n/2;
A = sparse(size(D,2),size(X,2));        %初始化系数矩阵
for k=1:1:P,                            
    a=[];
    x=X(:,k);                           %对信号逐列表示
    residual=x;
	indx = [];
	a = [];
	currResNorm2 = sum(residual.^2);
	j = 0;
    while currResNorm2>E2 & j < maxNumCoef,    %最大索引数目小于n/2
		j = j+1;
        proj=D'*residual;
        pos=find(abs(proj)==max(abs(proj)));
        pos=pos(1);                     %选取出投影绝对值最大的一个，若有多个，则选取第一个
        indx(j)=pos;
        a=pinv(D(:,indx(1:j)))*x;       %求伪逆来与单列信号相乘得到系数
        residual=x-D(:,indx(1:j))*a;
		currResNorm2 = sum(residual.^2);
   end;
   if (length(indx)>0)
       A(indx,k)=a;                     %逐列更新系数矩阵A
   end
end;
return;
