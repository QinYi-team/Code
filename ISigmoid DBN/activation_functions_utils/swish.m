% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
function  X = swish(A,nn)
belta = 10;
temp = 1./(1+exp(-belta *A));
X = A .* temp;



