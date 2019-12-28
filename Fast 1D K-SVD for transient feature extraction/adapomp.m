function [s3] = adapomp(D1,s,c)
% s:parameter of overcomplete dictionary
% c:Parameter for stopping criterion 
% =========================================================================
%                          Written by Yi Qin
% =========================================================================
for i=1:100
[A1]=OMP(D1,s,i);     %运用OMP函数
a = abs(A1(A1~=0));
b=max(a);
d=min(a);
if b>c*d
    break;
end
end
ss=D1*A1;
s3=s-ss;              %分离谐波和调制信号