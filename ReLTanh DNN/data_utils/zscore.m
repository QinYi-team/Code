% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
function [x, mu, sigma] = zscore(x)
    mu=mean(x);	    % 均值
    sigma=max(std(x),eps);   % std(x)按列求出每一列的标准差;   eps(x)求出abs（x）到最近一个浮点数的证书距离,  默认：eps=eps（1）=2.2204e-16    ； 这里的max就是找出eps矩阵和实体店矩阵中的最大值，目的是将std中的0替换为eps（1），因为0不能再充当分母
	x=bsxfun(@minus,x,mu);  % bsxfun（fun,A,B）对两个矩阵A和B之间的每一个元素进行指定的计算（函数fun指定）；   minus（A,B）=  A -B
	x=bsxfun(@rdivide,x,sigma);   % rdivide(A,B) = A ./ B   实现点除功能
end

% 此处用的归一化公式为： （x-x平均值）/std
