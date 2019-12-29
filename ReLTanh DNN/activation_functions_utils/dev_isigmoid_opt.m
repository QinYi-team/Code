% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数——isigmoid_opt的导数（相比于isigmoid，isigmoid_opt用阈值处sigmoid的导数作为之后的斜率）
function X = dev_isigmoid_opt(A,nn)

thp = nn.opts.th_god;      % 正数方向的阈值
thn = -nn.opts.th_god;     % 负数方向的阈值
kp  = nn.opts.k_god;       % 正数方向的斜率
kn  = nn.opts.k_god;       % 负数方向的斜率
X  = zeros(size(A));       % 与A同样size的矩阵，用于存储计算得到的矩阵

% 小于thn的部分
idx    = find(A<thn);                          % A中小于thn的元素的坐标（该坐标按照从上到下，从左到右的顺序依次检索）
X(idx) = sigmoid(thn).*(1- sigmoid(thn));      % 小于thn的部分，用sigmoid在thn处的导数作为斜率

% thn与thp之间的部分
idx    = find(A<=thp & A>=thn);     % thn与thp之间的部分按照正常sigmoid计算导数
t(idx) = 1./(1 + exp(-A(idx)));
X(idx) = t(idx).*(1- t(idx));

% 大于thp的部分
idx    = find(A>thp);                          % 大于thp的部分，用sigmoid在thp处的导数作为斜率
X(idx) = sigmoid(thp).*(1- sigmoid(thp));
