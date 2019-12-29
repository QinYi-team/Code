% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 子函数——前馈运算
% the feedfward process of NN
function nn = nnff(nn, x, y)                % 调用格式： nn = nnff(nn, batch_x, batch_y);

    numlayer  = numel(nn.opts.netsize);     % 网络层数
    numsample = size(x, 1);                 % 传进来的这一个batch的样本数
    
    
    % 输入层前馈
    nn.net{1}.in    = x;                    % 网络第一层输入值、净输入值输出值都是x，实际上没有对它进行计算
    nn.net{1}.netin = x;
    nn.net{1}.out   = x;                    % 将输入数据直接传递给第一层作为其初始数据

    % 隐含层前馈
    for i = 2 : numlayer-1        %  从第二层开始，涉及到传递函数变化，顶层是分类器因此不再这里计算
        
        nn.net{i}.in    = nn.net{i-1}.out;     % 本层的输入值是上一层的输出值
        nn.net{i}.netin = nn.net{i}.in * nn.net{i}.w + repmat(nn.net{i}.c , numsample ,1);        % 计算净输入值 
        
        switch nn.opts.activation_function                            % 判断不同的传递函数
            case 'sigmoid'                                            % 普通的sigmoid，
                nn.net{i}.out = sigmoid(nn.net{i}.netin);             % 这里相当于是sigm（net+b）
            case 'sigmoid_time'                                       % 普通sigmoid的功能，与isigmoid、hisigmoid等相同的程序结构，用于比较时间差，剔除因为程序写法导致的时间干扰
                nn.net{i}.out = sigmoid_time(nn.net{i}.netin , nn);
            case 'tanh_opt'                                           % 优化的tanh，普通的tanh输出是[-1,1],tanh_opt是[0,1]
                nn.net{i}.out = tanh_opt(nn.net{i}.netin);            
            case 'tanh'                                               % 普通的tanh     
                nn.net{i}.out = tanh(nn.net{i}.netin);                % tanh是系统函数，可直接调用
            case 'itanh'                                              % itanh     
                nn.net{i}.out = itanh(nn.net{i}.netin , nn);               % tanh是系统函数，可直接调用   
            case 'itanh_opt'                                              % itanh     
                nn.net{i}.out = itanh_opt(nn.net{i}.netin , nn);               % tanh是系统函数，可直接调用                     
            case 'isigmoid'                                           % 将sigmoid在两端的特定阈值外的函数改成了直线，直线的斜率是预设的
                nn.net{i}.out = isigmoid(nn.net{i}.netin, nn); 
            case 'isigmoid_opt'                                       % 将sigmoid在两端的特定阈值外的函数改成了直线，直线斜率是sigmoid在阈值处的导数
                nn.net{i}.out = isigmoid_opt(nn.net{i}.netin, nn); 
            case 'hisigmoid'                                          % 半边激活的isigmoid，也就是负数区域为普通sigmoid函数，正数区域大于阈值后的部分改为直线，直线斜率是预设的
                nn.net{i}.out = hisigmoid(nn.net{i}.netin, nn);  
            case 'hisigmoid_opt'                                      % 半边激活的isigmoid，也就是负数区域为普通sigmoid函数，正数区域大于阈值后的部分改为直线，直线斜率是sigmoid在阈值处的导数
                nn.net{i}.out = hisigmoid_opt(nn.net{i}.netin, nn); 
            case 'ReLU'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
                nn.net{i}.out = ReLU(nn.net{i}.netin,nn);
            case 'LReLU'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
                nn.net{i}.out = LReLU(nn.net{i}.netin,nn);
            case 'PReLU'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
                nn.net{i}.out = PReLU(nn.net{i}.netin,nn,i);                  
            case 'ELU'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
                nn.net{i}.out = ELU(nn.net{i}.netin,nn);          
            case 'reltanh_opt'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
                nn.net{i}.out = reltanh_opt(nn.net{i}.netin,nn,i);  
            case 'hreltanh_opt'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
                nn.net{i}.out = hreltanh_opt(nn.net{i}.netin,nn,i);         
            case 'swish'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
                nn.net{i}.out = swish(nn.net{i}.netin,nn);     
            case 'softplus'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
                nn.net{i}.out = softplus(nn.net{i}.netin,nn);                   
            case 'hexpo'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
                nn.net{i}.out = hexpo(nn.net{i}.netin,nn);                         
        end
%{       
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
        nn.a{i} = [ones(num_samples,1) nn.a{i}];    % 加入偏置值基列，用来在下一层中使用
        nn.netinput{i} = [ones(num_samples,1) nn.netinput{i}];
 %}
    end
%% 倒数第二层单独处理
%{
    for i = numlayer-1        %  导数第二层 
        nn.net{i}.in    = nn.net{i-1}.out;     % 本层的输入值是上一层的输出值
        nn.net{i}.netin = nn.net{i}.in * nn.net{i}.w + repmat(nn.net{i}.c , numsample ,1);        % 计算净输入值 
        nn.net{i}.out = tanh(nn.net{i}.netin);                % tanh是系统函数，可直接调用
    end
%}
%%     
    % 输出层前馈
    nn.net{end}.in    = nn.net{end - 1}.out;      % 输入值是前一层的输出值
    nn.net{end}.netin = nn.net{end}.in * nn.net{end}.w + repmat(nn.net{end}.c , numsample ,1);      % 净输入值
  
    switch nn.opts.output_function                             % 根据不同的输出层分类器，确定输出值计算方式
        case 'sigmoid'                                         % sigmoid分类器
            nn.net{end}.out = sigmoid(nn.net{end}.netin);
        case 'tanh'                                         % sigmoid分类器
            nn.net{end}.out = tanh(nn.net{end}.netin);    
        case 'linear'                                          % 线性分类器
            nn.net{end}.out = nn.net{end}.netin;
        case 'ReLU'                                          % 线性分类器
            nn.net{end}.out = ReLU(nn.net{end}.netin,nn);
        case 'LReLU'                                          % 线性分类器
            nn.net{end}.out = LReLU(nn.net{end}.netin,nn);
        case 'PReLU'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
            nn.net{end}.out = PReLU(nn.net{end}.netin,nn,i+1);      
        case 'ELU'                                               % 此处的ReLU实际上是LReLU，即负数区域有一个较小的斜率
            nn.net{end}.out = ELU(nn.net{end}.netin,nn);                    
        case 'reltanh_opt'                                          % 线性分类器
            nn.net{end}.out = reltanh_opt(nn.net{end}.netin,nn,i+1);
        case 'hreltanh_opt'                                          % 线性分类器
            nn.net{end}.out = hreltanh_opt(nn.net{end}.netin,nn,i+1);        
        case 'swish'                                          % 线性分类器
            nn.net{end}.out = swish(nn.net{end}.netin,nn);          
        case 'softplus'                                          % 线性分类器
            nn.net{end}.out = softplus(nn.net{end}.netin,nn);                   
        case 'hexpo'                                          % 线性分类器
            nn.net{end}.out = hexpo(nn.net{end}.netin,nn);                            
        case 'softmax'                                         % softmax分类器，MATLAB给出的形式
            temp = exp(bsxfun(@minus, nn.net{end}.netin, max(nn.net{end}.netin,[],2)));      % max(nn.a{n},[],2)找出每一行的最大值     
            nn.net{end}.out = bsxfun(@rdivide, temp, sum(temp, 2)); 
        case 'softmax_st'                                      % softmax_st分类器，自己理解的形式，softmax(n)=exp(n)/sum(exp(n))   
            temp = exp(nn.net{end}.netin);
            nn.net{end}.out = bsxfun(@rdivide, temp, sum(temp, 2));     % 将sum(y,2)复制size(y,2列
      
    end

    % 真实误差与性能误差
    nn.net{end}.err = y - nn.net{end}.out;           % 真实误差，用于nnff中计算更新量
    switch nn.opts.output_function
        case {'sigmoid','tanh', 'linear','PReLU','ReLU','LReLU','reltanh_opt','hreltanh_opt','ELU','hexpo','softplus'}
            nn.loss = 1/2 * sum(sum(nn.net{end}.err .^ 2)) / numsample;     % numsample是这个batch中的样本数
        case 'softmax'
            nn.loss = -sum(sum(y .* log(nn.net{end}.out))) / numsample;     % 未知
        case 'softmax_st'
            nn.loss = -sum(sum(y .* log(nn.net{end}.out))) / numsample;
        otherwise
            nn.loss = 1/2 * sum(sum(nn.net{end}.err .^ 2)) / numsample;     % numsample是这个batch中的样本数
    end

