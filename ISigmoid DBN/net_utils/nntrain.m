% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 子函数——nn训练
% the nn are trained by the programs
function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)


assert(isfloat(train_x), 'train_x must be a float');    % isfloat(train_x)判断train_x是不是float类型，assert（）当条件不满足时就会报错
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')       % 必须是4个或者6个输入数据

loss.train.e               = [];          % 这几个参数尚未启用，待改进功能后使用
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
loss_old  = inf;      % loss_old用来记录loss 变化，以便调节学习速率

% 判断是否进行验证
opts.validation = 0;        % 如果opts.validation = 0就表示不启用验证功能
if nargin == 6              % 如果子函数从上一层函数中承袭了6个变量，那就意味i里面含有验证样本集
    opts.validation = 1;    % 如果从上一层函数传入了验证样本集，就将opts.validation = 1 表示启用验证
end

% 判断是否画出相关图形
fhandle = [];                                 % 设置空句柄
if isfield(opts,'plot') && opts.plot == 1     % isfield(S,FIELDNAMES) 用于判断FIELDNAMES变量是否在S这个结构体里面，  这里的意思是要求必须提前设定了plot变量并且变量值为1， 才会执行画图
    fhandle = figure();                       % 满足条件就将figure()赋值给空句柄，用于调用
end

% 训练参数
numsamples = size(train_x, 1);     % 训练样本数
batchsize  = opts.batchsize;       % 小批量数据集规模
numepochs  = opts.numepochs;       % 循环次数
numbatches = numsamples / batchsize;                                 % batch数目
assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');     % 要求设置的batchsize必须能够整除总的训练样本集 这里应该可以改成允许除不尽，然后把多余的去除

% 设计空集，用于后期存储
L = zeros(numepochs*numbatches,1);     % 每一次训练，每一个batch的误差

tracking.loss_train     = [];          % 跟踪记录训练误差变化
tracking.accuracy_train = [];          % 跟踪记录训练错误率变化
tracking.fail_train     = [];          % 只需要保存最后一次的错误
tracking.thp            = [];
tracking.thn            = [];
for i = 1 : length(val_y)            % 依次检索每个训练集
    tracking.loss_val{i}     = [];          % 跟踪记录验证误差变化
    tracking.accuracy_val{i} = [];          % 跟踪记录验证错误率变化
    tracking.fail_val{i}     = [];          % 只需要保存最后一次的错误
end
tracking.lr = [];             % 记录学习速率
tracking.time_train = [];     % 记录训练时间，把每个循环训练时间来及起来， 其他画图等时间不计算进去

% 正式的训练
for i = 1 : numepochs       % 大循环，依次进行每一次学习
    tic;                    % 计时起点
    
    % randbatch = randperm(numsamples);
    randbatch = nn.opts.randbatch;          % nnmain中预设，用于随机组合出batch
    for l = 1 : numbatches                  % 小循环，依次检索batch
        batch_x = train_x(randbatch((l - 1) * batchsize + 1 : l * batchsize), :);     % 现场组装出batch，而不提前做·
        batch_y = train_y(randbatch((l - 1) * batchsize + 1 : l * batchsize), :);
        
        % 判断是否加入噪声
        %{ 
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        %}
        
        % 核心步骤
        nn = nnff(nn, batch_x, batch_y);       % 调用nnff，前馈运算
        nn = nnbp(nn,i);                       % 调用nnbp，反向调节，计算权值、偏置值的更新量，并完成更新
        %nn = nnapplygrads(nn);                % 完成更新（该功能已被融合到了nnbp中）
        
        L(end + 1) = nn.loss;                  % 记录每一次循环，每一个batch的误差均方根
        
    end
    t = toc;                                   % 计时终止，次数计时实际上只计算了训练过程中nnff和nnbp的时间，而将其他共有的时间去除
    tracking.time_train(end + 1) = t;          % 将每一次训练的时间存起来
    
    % 确定是否启用验证
    if opts.validation == 1                    % 如果有传入验证样本集，就进行用验证集验证
        tracking = nntracking(nn, tracking, train_x, train_y, val_x, val_y);   % 这里没有进行batch
        str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', tracking.accuracy_train(end) , tracking.accuracy_val{1}(end));    % 每个学习epoch结束都输出相关结果，以观察进度
    else                                       % 如果没有验证样本集，就只计算训练样本集
        tracking = nntracking(nn, tracking, train_x, train_y);                 % 这里没有进行batch
        str_perf = sprintf('; Full-batch train mse = %f', tracking.accuracy_train(end));
    end
    
    % 画图功能
%     if ishandle(fhandle)   % ishandle（）如果括号中使图形处理句柄，就会返回true
%         nnupdatefigures(nn, fhandle, loss, opts, i);
%     end
    
    ave_loss = mean(L((end-numbatches+1):(end)));           % 计算本次循环的所有batch的平均误差，可以考虑将这个误差找直接换成tracking函数中的误差
    disp(['epoch F' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(ave_loss) str_perf]);
    
    if  tracking.loss_val{1}(end) > loss_old                   % 如果本次学习的验证误差大于上一次的误差，那么就减小学习速率；loss_old初始值为inf，因此第一次是不会更新学习速率的
        nn.opts.lr   = nn.opts.lr * nn.opts.lr_adjust;      % 执行学习速率更新  
    end
    loss_old = tracking.loss_val{1}(end);                      % 将本次的验证误差赋值给loss_old用于下一次比较
    tracking.lr(end + 1) = nn.opts.lr;                      % 将每一次学习速率存起来，用于观察学习速率变化情况
end  

nn.tracking = tracking;                                     % 便于nn结构体整体传出子函数

