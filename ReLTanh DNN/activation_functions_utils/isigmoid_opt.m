% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数——isigmoid_opt（相比于isigmoid，isigmoid_opt将预设斜率换成阈值出的sigmoid的导数）
function X = isigmoid_opt(A, nn)

thp = nn.opts.th_god;      % 正数方向的阈值
thn = -nn.opts.th_god;     % 负数方向的阈值
kp  = nn.opts.k_god;       % 正数方向的斜率
kn  = nn.opts.k_god;       % 负数方向的斜率
X  = zeros(size(A));       % 与A同样size的矩阵，用于存储计算得到的矩阵

% 小于thn的部分
idx    = find(A<thn);                      % A中小于thn的元素的坐标（该坐标按照从上到下，从左到右的顺序依次检索）
t = 1./(1 + exp(-thn));                    % 临时参数，计算thp处的sigmoid函数值，以计算该出的sigmoid的导数
X(idx) = t + (A(idx)-thn)*(t.*(1- t));     % 小于thn部分的实际输出等于阈值的sigmoid函数输出值 + 直线部分对应的值

% thn与thp之间的部分
idx    = find(A<=thp & A>=thn);  
X(idx) = 1./(1 + exp(-A(idx)));            % sigmoid函数输出值

% 大于thp的部分
idx    = find(A>thp);  
t = 1./(1 + exp(-thp));
X(idx) = t + (A(idx)-thp)*(t.*(1- t));     % 大于thp部分的实际输出等于阈值的sigmoid函数输出值 + 直线部分对应的值
