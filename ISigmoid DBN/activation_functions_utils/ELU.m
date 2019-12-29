% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
function  X = ELU(A,nn)
  
alpha = 1;


X  = zeros(size(A));

idx    = find(A>=0);   
X(idx) = A(idx) * 1;

idx    = find(A<0);   
X(idx) = exp(A(idx)) * alpha - alpha;


