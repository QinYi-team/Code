% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
function  X = hexpo(A,nn)

a = 1;
b = 1;
c = 1;
d = 1;

a = 10;
b = 10;
c = 1;
d = 20;

X  = zeros(size(A));

idx    = find(A>=0); 
temp = exp(-A / b);
X(idx) = -a *  (temp(idx) - 1);

idx    = find(A<0); 
temp = exp(A / d);
X(idx) = c * (temp(idx) - 1);


