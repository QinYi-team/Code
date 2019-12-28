function [S] = Hankel_matrix(s3,l,m,n,N)
%生成Hankel矩阵 Generate Hankel matrix
% =========================================================================
%                          Written by Yi Qin
% =========================================================================
s4=reshape(s3,l,N/l);                                   %对信号进行分段

%S = zeros(m,n);
for k=1:N/l
    for i=1:m
        for j=1:n
            S(i,j,k)=s4(i+j-1,k);                        %构造Hankel矩阵
        end
    end
end
