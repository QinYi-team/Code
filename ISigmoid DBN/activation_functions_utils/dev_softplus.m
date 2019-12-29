% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% the derivative of Softplus
function  X = dev_softplus(A,nn)

temp = exp(A);

X = temp ./ (1 + temp);



