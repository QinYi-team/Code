% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 子函数——建立SAE网络
function sae = saesetup(sae,rand_initial,hyperpara)
netsize = sae.opts.netsize;        % 网络的层数与每层的节点数
num_ae  = length(netsize)-2;       % SAE个数

% 各个自编码器
for i = 1 : num_ae        % 依次检索每层
    netsize_ae   = [netsize(i) netsize(i+1) netsize(i)];    % 此ae的结构
    sae.ae{i} = nnsetup(netsize_ae,rand_initial{i},hyperpara);        % 调用nnsetup建立自编码器
    sae.ae{i}.opts.netsize   = netsize_ae;                  % 此ae的结构
    sae.ae{i}.opts.numepochs = sae.opts.numepochs;          % RBM预训练循环次数
    sae.ae{i}.opts.batchsize = sae.opts.batchsize;          % 设置RBM预训练的batch的大小
    sae.ae{i}.opts.lr        = sae.opts.lr;                 % 设置RBM预训练的初始学习速率
    sae.ae{i}.opts.lr_adjust = sae.opts.lr_adjust;          % 设置RBM预训练的学习速率调整因子
    sae.ae{i}.opts.momentum  = sae.opts.momentum;           % 设置RBM预训练的动量项
    sae.ae{i}.opts.randbatch = sae.opts.randbatch;          % 用于saetrain中的batch的随机选择 
    
    sae.ae{i}.opts.output_function      = sae.opts.output_function;           % 设置sae网络顶层分类器
    sae.ae{i}.opts.activation_function  = sae.opts.activation_function;           % 设置sae网络的传递函数 
end


% 顶层classifier
dim         = [netsize(end-1),netsize(end)];      % 网络结构
temp        = sqrt( 3 / dim(1));
temp        = sqrt( 6 /(dim(1) + dim(2)));
sae.classifier.w = 2*(rand_initial{end}(1:dim(1),1:dim(2))- 0.5) * temp *2;  % ;
sae.classifier.c = zeros(1,dim(2));                % 偏置值初始化为0








