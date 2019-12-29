% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
function  X = LReLU(A,nn)
  
X  = zeros(size(A));

idx    = find(A>=0);   
X(idx) = A(idx) * 1;

idx    = find(A<0);   
X(idx) = A(idx) * 0.01;


