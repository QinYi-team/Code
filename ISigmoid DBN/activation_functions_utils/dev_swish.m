% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
% the derivative of Swish
function  X = dev_swish(A,nn)

belta = 10;
temp = 1./(1+exp(-belta * A));
X = temp .* (1 + belta * A - belta *A.*temp);


