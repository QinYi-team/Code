% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% the derivative of PReLU
function  nn = dev_PReLU(A,nn,i)

% 参数学习



X=zeros(size(A));       % 与A同样size的矩阵，用于存储计算得到的矩阵 

% 大于等于0的部分
idx    = find(A>=0);    % A中大于0的元素的坐标（该坐标按照从上到下，从左到右的顺序依次检索）
nn.net{i}.k_PReLU= nn.net{i}.k_PReLU + 0;    % K_PReLU不变
X(idx) = 1;             % 对这一部分直接将导数赋值为1

%% 小于0的部分

% 同层共享k_PReLU
% {
idx    = find(A<0);
temp = zeros(size(A));
temp(idx) = A(idx) ;
nn.net{i}.k_PReLU = nn.net{i}.k_PReLU - nn.opts.lr * mean(mean(temp .* (nn.net{i}.err))) ; 
X(idx) = nn.net{i}.k_PReLU;

nn.net{i}.d = X;
%}


% 每个节点的k_PReLU不同
%{
[row,col,val]    = find(A<0);     % 对净输入值小于0的这部分，将其到导数直接赋值为0.01
ttt = zeros(size(A));
ttt(row,col) = A(row,col);
nn.net{i}.k_PReLU = nn.net{i}.k_PReLU + 0.01 * mean(ttt,1) ;
k = nn.net{i}.k_PReLU; 
temp = repmat(k,size(A,1),1);
X(row,col) = temp(row,col) ;
%}


