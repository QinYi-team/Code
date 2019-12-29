% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 辅助代码

%% 计算净输入值的分布情况（将一下代码复制到command window中即可运算）
for i = 2 : numel(nn.opts.netsize)    % 依次检索一层网络；numel(nn.opts.netsize) 计算网络层数
    for j = 1 : 10        % j用来设置区间，用来将净输入值的绝对值划分为多个区间
        t(i,j) = numel(find(abs(nn.net{i}.netin) > j)) / numel(nn.net{i}.netin);     % 分别计算落入不同区间的净输入值的绝对值所占的总的净输入值个数的比例
    end
end


%% 计算sigmoid在各处的斜率（将一下代码复制到command window中即可运算）
for i=1:10                
    y = 1/(1+exp(-i));
    dev(i) =(1- y)*y;
end
        