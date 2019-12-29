% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数——建立nn网络（如果选择网络种类是BPNN，就直接用此函数构建网络）
% the NN net are constructed as follows
function nn = nnsetup(netsize,rand_initial,hyperpara)      % 调用格式为：nn = nnsetup([784 20 10]);    

% 相关设置
nn.size   = netsize;          % 网络结构
nn.n      = numel(nn.size);   % 得到网络的层数

nn.opts.activation_function              = 'NULL';   %  传递函数
nn.opts.lr                               = 1;         %  初始学习速率
nn.opts.momentum                         = 0;            %  动量项
nn.opts.lr_adjust                        = 0.95;         %  学习速率调整因子
nn.opts.weightPenaltyL2                  = 0;            %  权值惩罚正则化因子
nn.opts.nonSparsityPenalty               = 0;            %  稀疏惩罚
nn.opts.sparsityTarget                   = 0;            %  Sparsity target
nn.opts.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
nn.opts.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
nn.opts.testing                          = 0;            %  Internal variable. nntest sets this to one.
nn.opts.output_function                  = 'NULL';    %  输出层分类器

% 开始
nn.net{1}.w = [0];         % 输入层 
nn.net{1}.c = [0];

for i = 2 : nn.n % 隐含层与输出层

    % xaviar归一化
    % {
    dim         = [netsize(i-1),netsize(i)];      % 网络结构
    temp        = sqrt( 3 / dim(1));
    temp        = sqrt( 6 /(dim(1) + dim(2)));
    nn.net{i}.w = 2*(rand_initial(1:dim(1),1:dim(2))- 0.5) * temp;  % ;
    nn.net{i}.c = zeros(1,dim(2));                 % 偏置值初始化为0
    %}
     
    nn.net{i}.k_PReLU = hyperpara.k_PReLU ;  %*ones(1,dim(2)) ;
    nn.net{i}.thp = hyperpara.thp ;          %*ones(1,dim(2)) ;
    nn.net{i}.thn = hyperpara.thn;           %*ones(1,dim(2)) ;
     
end

