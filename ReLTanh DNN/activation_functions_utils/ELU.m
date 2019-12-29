% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
function  X = ELU(A,nn)
  
alpha = 1;


X  = zeros(size(A));

idx    = find(A>=0);   
X(idx) = A(idx) * 1;

idx    = find(A<0);   
X(idx) = exp(A(idx)) * alpha - alpha;


