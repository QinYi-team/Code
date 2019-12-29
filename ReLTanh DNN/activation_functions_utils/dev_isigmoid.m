% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数——isigmoid的导数
function X = dev_isigmoid(A,nn)

thp = nn.opts.th_god;      % 正数方向的阈值
thn = -nn.opts.th_god;     % 负数方向的阈值
kp  = nn.opts.k_god;       % 正数方向的斜率
kn  = nn.opts.k_god;       % 负数方向的斜率
X  = zeros(size(A));       % 与A同样size的矩阵，用于存储计算得到的矩阵

% 小于thn的部分
idx    = find(A<thn);       % A中小于thn的元素的坐标（该坐标按照从上到下，从左到右的顺序依次检索）
X(idx) = kn;                % 小于thn的部分，用预设的kp作为导数

% thn与thp之间的部分
idx    = find(A<=thp & A>=thn);   % thn与thp之间的部分按照正常sigmoid计算导数
t(idx) = 1./(1 + exp(-A(idx)));
X(idx) = t(idx).*(1- t(idx));

% 大于thp的部分
idx    = find(A>thp);        % 大于thp的部分，用预设的kp作为导数
X(idx) = kp;               


%}

%{
function X = dev_isigmoid(A,nn)

th = 7;
kp = 0.01;
kn = 0.01;
X  = zeros(size(A));

idx    = find(A<-th);   
X(idx) = A(idx).*(1- A(idx));

idx    = find(abs(A)<=th);  
X(idx) = A(idx).*(1- A(idx));

idx    = find(A>th);  
X(idx) = A(idx).*(1- A(idx));
%}

%{
function X = dev_isigmoid(A)

th = 7;
kp = 0.01;
kn = 0.01;
X  = zeros(size(A));

idx    = find(A<-th);   
X(idx) = A(idx).*(1- A(idx));

idx    = find(abs(A)<=th);  
X(idx) = A(idx).*(1- A(idx));

idx    = find(A>th);  
X(idx) = A(idx).*(1- A(idx));
%}