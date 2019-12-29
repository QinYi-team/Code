% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
function  X = PReLU(A,nn,i)

%% 参数学习

k = nn.net{i}.k_PReLU;     

X  = zeros(size(A));

idx    = find(A>=0);   
X(idx) = A(idx) * 1;

%%  小于0的部分
% 同层共享k_PReLU
% {
idx    = find(A<0);   
X(idx) = A(idx) .* k;
%}




% 每个节点的k_PReLU不同
%{
[row,col,val]    = find(A<0);     % 对净输入值小于0的这部分，将其到导数直接赋值为0.01
temp = A .* repmat(k,size(A,1),1);
X(row,col) = temp(row,col);
%}

