% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数——nn反向调节运算，计算权值、偏置值的更新量
%% the Backpropagation process for NN
function nn = nnbp(nn,time)        % time是nnmain中当前训练次序

numlayer      = numel(nn.opts.netsize);     % 网络层数   % 层数，输入层是第一层
sparsityError = 0;                          % 稀疏误差
batchsize     = nn.opts.batchsize;          % batch的大小

%% 矫正斜率与矫正误差
%  输出层
switch nn.opts.output_function                                         % 根据不同的输出层分类器确定矫正斜率的计算方式
    case 'sigmoid'
        nn.net{end}.d = nn.net{end}.out .* (1 - nn.net{end}.out);      % 矫正斜率为对应的导数
    case 'tanh'
        nn.net{end}.d   = dev_tanh(nn.net{end}.netin);                   % 矫正斜率为对应的导数   
    case {'softmax','softmax_st','linear'}
        nn.net{end}.d = ones(size(nn.net{end}.out));                   % 矫正斜率为1
    case {'ReLU'}
        nn.net{end}.d = dev_ReLU(nn.net{end}.netin,nn);                   % 矫正斜率为1   
    case {'LReLU'}
        nn.net{end}.d = dev_LReLU(nn.net{end}.netin,nn);                   % 矫正斜率为1 
    case {'PReLU'}
        nn = dev_PReLU(nn.net{end}.netin,nn,numlayer);                   % 矫正斜率为1     
    case 'ELU'
        nn.net{end}.d = dev_ELU(nn.net{end}.netin,nn);                   % 矫正斜率为对应的导数        
    case {'reltanh_opt'}
        nn = dev_reltanh_opt(nn.net{end}.netin,nn,numlayer);                   % 矫正斜率为1           
    case {'hreltanh_opt'}
        nn = dev_hreltanh_opt(nn.net{end}.netin,nn,numlayer);                   % 矫正斜率为1    
    case {'swish'}
        nn.net{end}.d = dev_swish(nn.net{end}.netin,nn);                   % 矫正斜率为1   
    case {'softplus'}
        nn.net{end}.d = dev_softplus(nn.net{end}.netin,nn);                   % 矫正斜率为1          
    case {'hexpo'}
        nn.net{end}.d = dev_hexpo(nn.net{end}.netin,nn);                   % 矫正斜率为1            
end
nn.net{end}.dd = nn.net{end}.err .* nn.net{end}.d;                     % 真实误差与矫正斜率计算出校正误差

% 隐含层
nn.net{end-1}.err = nn.net{end}.dd * nn.net{end}.w';                   % 输出层前一层绝对误差

% 倒数第二层单独处理
%{
for i = numlayer-1        %  导数第二层 
    nn.net{i}.d = dev_tanh(nn.net{i}.netin);                   % 矫正斜率为对应的导数
end
nn.net{i}.dd    = nn.net{i}.err .* nn.net{i}.d;                    % 真实误差与矫正斜率计算出校正误差
nn.net{i-1}.err = nn.net{i}.dd * nn.net{i}.w';                     % 计算前一层的绝对误差
%}
for i = (numlayer -1) : -1 : 2                                        % 依次检索每个隐含层
    switch nn.opts.activation_function                                 % 根据不同的传递函数确定校正误差计算方式
        case 'sigmoid'
            nn.net{i}.d = nn.net{i}.out .* (1 - nn.net{i}.out);        % 矫正斜率为对应的导数，
        case 'sigmoid_time'
            nn.net{i}.d = dev_sigmoid_time(nn.net{i}.netin , nn);      % 矫正斜率为对应的导数
        case 'tanh_opt'
            nn.net{i}.d = dev_tanh_opt(nn.net{i}.netin);               % 矫正斜率为对应的导数
        case 'tanh'
            nn.net{i}.d = dev_tanh(nn.net{i}.netin);                   % 矫正斜率为对应的导数
        case 'itanh'
            nn.net{i}.d = dev_itanh(nn.net{i}.netin , nn);             % 矫正斜率为对应的导数
        case 'itanh_opt'
            nn.net{i}.d = dev_itanh_opt(nn.net{i}.netin , nn);         % 矫正斜率为对应的导数            
        case 'isigmoid'
            nn.net{i}.d = dev_isigmoid(nn.net{i}.netin , nn);          % 矫正斜率为对应的导数
        case 'isigmoid_opt'
            nn.net{i}.d = dev_isigmoid_opt(nn.net{i}.netin , nn);      % 矫正斜率为对应的导数
        case 'hisigmoid'
            nn.net{i}.d = dev_hisigmoid(nn.net{i}.netin , nn);         % 矫正斜率为对应的导数
        case 'hisigmoid_opt'
            nn.net{i}.d = dev_hisigmoid_opt(nn.net{i}.netin , nn);     % 矫正斜率为对应的导数
        case 'ReLU'
            nn.net{i}.d = dev_ReLU(nn.net{i}.netin,nn);                   % 矫正斜率为对应的导数
        case 'LReLU'
            nn.net{i}.d = dev_LReLU(nn.net{i}.netin,nn);                   % 矫正斜率为对应的导数            
        case 'PReLU'
            nn = dev_PReLU(nn.net{i}.netin, nn ,i);                   % 矫正斜率为对应的导数
        case 'ELU'
            nn.net{i}.d = dev_ELU(nn.net{i}.netin,nn);                   % 矫正斜率为对应的导数
        case 'reltanh_opt'
            nn = dev_reltanh_opt(nn.net{i}.netin,nn,i);                   % 矫正斜率为对应的导数
        case 'hreltanh_opt'
            nn = dev_hreltanh_opt(nn.net{i}.netin,nn,i);                   % 矫正斜率为对应的导数
        case 'swish'
            nn.net{i}.d = dev_swish(nn.net{i}.netin,nn);                   % 矫正斜率为对应的导数  
        case 'softplus'
            nn.net{i}.d = dev_softplus(nn.net{i}.netin,nn);                   % 矫正斜率为对应的导数               
        case 'hexpo'
            nn.net{i}.d = dev_hexpo(nn.net{i}.netin,nn);                   % 矫正斜率为对应的导数             
    end
    nn.net{i}.dd    = nn.net{i}.err .* nn.net{i}.d;                    % 真实误差与矫正斜率计算出校正误差
    nn.net{i-1}.err = nn.net{i}.dd * nn.net{i}.w';                     % 计算前一层的绝对误差
end

%% 权值与偏置值更新量
for i = numlayer : -1 : 2                                     % 倒序依次检索每一层，只检索到第二层，因为第一层是个装饰，权值与偏置值都是0，不起作用
    % 修正值
    nn.net{i}.ddw = nn.net{i}.in' * nn.net{i}.dd;             % 输入值与校正误差计算出权值更新量   % 这一步隐含了同一个batch中，所有权值相加，因此后面要对其除batchsize，求平均
    nn.net{i}.ddc = nn.net{i}.dd;                             % 偏置值更新量就只校正误差
    
    % 加上学习速率
    nn.net{i}.dddw = nn.opts.lr * nn.net{i}.ddw;              % 加上学习速率
    nn.net{i}.dddc = nn.opts.lr * nn.net{i}.ddc;
    
    % 平均修正值
    nn.net{i}.ddddw = nn.net{i}.dddw  / batchsize;             % 计算权值更新量的平均值
    nn.net{i}.ddddc = sum(nn.net{i}.dddc ,1) / batchsize;     % 偏置值的平均值，此处sum(nn.net{i}.dddc ,1)将batch中的所有sanple计算得到的偏置值更新量进行求和，以求出平均值，因为偏置值是列向量，而权值是矩阵
    
    % 加上动量项
    if time == 1                % 如果是第一次训练，就不使用动量项，因为不存在上一次的更新值
        nn.net{i}.dddddw = nn.net{i}.ddddw + nn.opts.momentum * 0;        % 因为不存在前一次的修正量，因此*0
        nn.net{i}.dddddc = nn.net{i}.ddddc + nn.opts.momentum * 0;
    else                     
        nn.net{i}.dddddw = nn.net{i}.ddddw + nn.opts.momentum * nn.net{i}.dddddw;       % 赋值语句右边的nn.net{i}.dddddw是前一次的更新量  
        nn.net{i}.dddddc = nn.net{i}.ddddc + nn.opts.momentum * nn.net{i}.dddddc;
    end
    
    % 赋值调整
    nn.net{i}.w = nn.net{i}.w + nn.net{i}.dddddw;
    nn.net{i}.c = nn.net{i}.c + nn.net{i}.dddddc;
end





