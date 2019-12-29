% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 子函数——ReLTanh
function X = hreltanh_opt(A, nn,i)

thp = nn.net{i}.thp;       % 正数方向的阈值
thn = nn.net{i}.thn;     % 负数方向的阈值

X  = zeros(size(A));       % 与A同样size的矩阵，用于存储计算得到的矩阵

% 小于thn的部分
% {
idx    = find(A<thn);                       % A中小于thn的元素的坐标（该坐标按照从上到下，从左到右的顺序依次检索）
t      = tanh(thn);                          % 临时参数，计算thn处的tanh函数值，以计算该出的tanh的导数
X(idx) = tanh(thn) + (A(idx)-thn)*(1-t^2);       % 小于thn部分的实际输出等于阈值的sigmoid函数输出值 + 直线部分对应的值
%}

% thn与thp之间的部分
idx    = find(A<=thp & A>=thn );         %            
X(idx) = tanh(A(idx));                      % sigmoid函数输出值

% 大于thp的部分
idx    = find(A>thp);  
t      = tanh(thp);                          % 临时参数，计算thn处的tanh函数值，以计算该出的tanh的导数
X(idx) = tanh(thp) + (A(idx)-thp)*(1-t^2);       % 大于thp部分的实际输出等于阈值的sigmoid函数输出值 + 直线部分对应的值 




