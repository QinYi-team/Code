% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 子函数——ReLTanh's derivative
function nn = dev_hreltanh_opt(A, nn,i)


thp = nn.net{i}.thp;       % 正数方向的阈值
thn = nn.net{i}.thn;     % 负数方向的阈值
X  = zeros(size(A));       % 与A同样size的矩阵，用于存储计算得到的矩阵

%梯度准备
t = tanh(A);
tt = 1 - power(t,2);
ttt = -2 * t .* tt;

% {
% 小于thn的部分
idx    = find(A<thn);                               % A中小于thn的元素的坐标（该坐标按照从上到下，从左到右的顺序依次检索）
temp = ttt(idx);
if ~isempty(temp) & ~isempty(nn.net{i}.err(idx))
    thn = thn -  mean(mean(temp .* (nn.net{i}.err(idx))));  %
end
if thn >= -1.5
    thn = -1.5;
end
nn.net{i}.thn = thn;
X(idx) = 1-tanh(thn)^2;                           % 小于thn部分的实际输出等于阈值的sigmoid函数输出值 + 直线部分对应的值
%}

% thn与thp之间的部分
idx    = find(A<=thp& A>=thn);                % & A>=thn         
X(idx) = tt(idx);                     % sigmoid函数输出值


% 大于thp的部分
idx    = find(A>thp);  
% {
temp = ttt(idx);
if ~isempty(temp) & ~isempty(nn.net{i}.err(idx))
    thp = thp -  mean(mean(temp .* (nn.net{i}.err(idx)))) ;  %
end
if thp <= 0
    thp = 0;
end
if thp >0.5
    thp =0.5;
end
nn.net{i}.thp = thp;
X(idx) = 1-tanh(thp)^2;      % 大于thp部分的实际输出等于阈值的tanh函数输出值 + 直线部分对应的值 
%}
nn.net{i}.d = X;


