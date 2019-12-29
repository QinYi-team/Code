% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% the derivative of Softplus
function  X = dev_softplus(A,nn)

temp = exp(A);

X = temp ./ (1 + temp);



