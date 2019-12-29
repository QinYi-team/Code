% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数——isigmoid（相比于sigmoid，isigmoid将大于阈值之外的部分换成了固定斜率的直线）
function X = isigmoid(A, nn)

thp = nn.opts.th_god;      % 正数方向的阈值
thn = -nn.opts.th_god;     % 负数方向的阈值
kp  = nn.opts.k_god;       % 正数方向的斜率
kn  = nn.opts.k_god;       % 负数方向的斜率
X  = zeros(size(A));       % 与A同样size的矩阵，用于存储计算得到的矩阵

% 小于thn的部分
idx    = find(A<thn);                                % A中小于thn的元素的坐标（该坐标按照从上到下，从左到右的顺序依次检索）
X(idx) = 1./(1 + exp(-thn)) + (A(idx)-thn)*kn;       % 小于thn部分的实际输出等于阈值的sigmoid函数输出值 + 直线部分对应的值

% thn与thp之间的部分
idx    = find(A<=thp & A>=thn);                      
X(idx) = 1./(1 + exp(-A(idx)));                      % sigmoid函数输出值

% 大于thp的部分
idx    = find(A>thp);  
X(idx) = 1./(1 + exp(-thp)) + (A(idx)-thp)*kp;       % 大于thp部分的实际输出等于阈值的sigmoid函数输出值 + 直线部分对应的值 


%{
function X = isigmoid(A,nn)

thp = 7;
thn = -7;
kp = 0.01;
kn = 0.01;
X  = A;

idx    = find(A<thn);   
X(idx) = 1./(1 + exp(-A(idx)));

idx    = find(A<=thp & A>=thn);  
X(idx) = 1./(1 + exp(-A(idx)));

idx    = find(A>thp);  
X(idx) = 1./(1 + exp(-A(idx)));
%}

%{
function X = isigmoid(A)

thp = 7;
thn = -7;
kp = 0;
kn = 0;
X  = A;

idx    = find(A<thn);   
X(idx) = 1./(1 + exp(-A(idx)));

idx    = find(A<=thp & A>=thn);  
X(idx) = 1./(1 + exp(-A(idx)));

idx    = find(A>thp);  
X(idx) =  1./(1 + exp(-A(idx))) ;
%}