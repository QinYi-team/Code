% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数——RBM预训练
function rbm = rbmtrain(rbm, x, opts)

assert(isfloat(x), 'x must be a float');          % 如果传入的x不是float类型，就报错
%assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');           % 如果x没有完成归一化，就报错
numsamples = size(x, 1);                                       % 传入的样本数
numbatches = numsamples / opts.batchsize;                      % batch的数目
assert(rem(numbatches, 1) == 0, 'numbatches not integer');     % 如果batch的数目不是整数，就报错
errold = inf;        % 预设errold，用于和本次的err比较，以便调整学习速率

% 正式学习
for i = 1 : opts.numepochs            % 依次训练         
    %kk = randperm(numsamples);
    randbatch = opts.randbatch;       % 在nnmain中预设，用于batch的随机选择
    errsum = 0;                       % 预设误差和，再循环中用到
    for l = 1 : numbatches            % 依次检索每个batch
        batch = x(randbatch((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);   % 临时构造batch，临时构造更方便，但是一旦构造完成就保证构batch不会变化了，不然会使得训练曲线波折很大
        
        % 完成一次CD
        V1_d2v = batch;                                                               % 第一次可见层到隐含层
        H1_v2h = sigmrnd(V1_d2v * rbm.w + repmat(rbm.c , opts.batchsize, 1));         % 第一次隐含层返回可见层
        V2_h2v = sigmoid(H1_v2h * rbm.w'+ repmat(rbm.b , opts.batchsize, 1));         % 第二次可见层到隐含层
        H2_v2h = sigmoid(V2_h2v * rbm.w + repmat(rbm.c , opts.batchsize, 1));         % 第二次隐含层返回可见层

        % 每个batch的参数修正量
        dw_batch = V1_d2v'*H1_v2h - V2_h2v'*H2_v2h;        % 计算权值、偏置值修正值
        db_batch = sum(V1_d2v) - sum(V2_h2v);
        dc_batch = sum(H1_v2h) - sum(H2_v2h);
        % 每个样本的参数修正量
        dw_sample = dw_batch / opts.batchsize;             % 计算修正值的平均值
        db_sample = db_batch / opts.batchsize;
        dc_sample = dc_batch / opts.batchsize;            
        % 参数修正          
        rbm.vw = opts.momentum * rbm.vw + opts.lr * dw_sample;     % 加入动量项与学习速率的处理
        rbm.vb = opts.momentum * rbm.vb + opts.lr * db_sample;
        rbm.vc = opts.momentum * rbm.vc + opts.lr * dc_sample;

        rbm.w = rbm.w + rbm.vw;      % 赋值生效
        rbm.b = rbm.b + rbm.vb;
        rbm.c = rbm.c + rbm.vc;

        err = sum(sum((V1_d2v - V2_h2v).^2)) / opts.batchsize;       % 计算重构误差
        errsum = errsum + err;                 % 累计本次学习中所有batch的重构误差和
    end

    rbm.recon_err(i) = errsum / numbatches;   % 计算本次学习的所有batch误差和的平均值 
    if i>=2       % 如果不是第一次学习，就要比较本次和前一次的重构误差，已决定是否调整学习速率
        if rbm.recon_err(i) > rbm.recon_err(i-1)  
           opts.lr = opts.lr_adjust * opts.lr;
        end
    end

    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(rbm.recon_err(i))]);      % 输出当前是第几次学习，以及当前的重构误差

end

