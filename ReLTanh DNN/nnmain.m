% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================

%% main function

%% 子函数与主函数切换
% 可修改为子函数，接受godfinger调用
function nn = nnmain(acfun,i_rep, flag,i_acfun,nettype,filename)          % 比较不同激活函数            

%% setup before training

scalerpool = {'minmax','maxabs','z-score'};
scaler = scalerpool{3};
prelr  = 0.005;                 % lr for pretrain
premom = 0.005;                 % momentum for pretrain
prelr_adjust = 0.99;            % lr can be adjusted by prelr_adjust
lr     = 0.001;                 % lr for train
mom    = 0.001;                 % mom for train
lr_adjust = 0.99;               % lr_adjust for train

% default parameters
if ~exist('flag','var')||isempty(flag)        flag = 'acfuns';   end       % if get no or empty flag, flag is set as 'acfuns' for default；
if ~exist('acfun','var')||isempty(acfun)      acfun = 'sigmoid'; end       % ditto；
if ~exist('nettype','var')||isempty(nettype)  nettype = 'DBN';   end       % ditto；
if ~exist('i_acfun','var')||isempty(i_acfun)  i_acfun = 1;       end       % ditto；
if ~exist('i_rep','var')||isempty(i_rep)      i_rep = 0;         end       % ditto；




%% 随机集权设置  random numbers are carete and load for initialization of weights and biases
% { 
for i = 1: 2500   rands{i} = randperm(i*10);         end     % 用于样本选择与minibatch
for i = 1: 20     rand_initial{i} = rand(100,100);   end     % 用于每一层权值初始

% {
%% load and slice up 
%%%%%%%%%% planetary gearbox data %%%%%%%%%%%%%

load ('gear_fault_featuredata_5120','featuredata');           % 加载故障数据   the dataset are loaded 
N      = 4000;                                                % the number of samples
randt  = rands{N/10};
dim    = [1:50];                                              % 特征选择   only the 
data   = double(featuredata.feature(randt(1:N),dim));               
label  = data_binarize(featuredata.labelload(randt(1:N),1)-1);  % 标签二值化处理，外层减1的原因是binarize函数内部有一个配合mnist的加1操作
load1  = featuredata.labelload(randt(1:N),2);                   % 提取负载数据



%% 选择训练数据及与验证数据集如何分配  training and testing samples 
% {
N  = size(label,1);              % number of samples
Nt = 2000;                       % number of samples for trianing
Nv = 250;                        % number of samples for validation
tx0 = data(1:Nt,:);              % 训练样本  the training samples
ty  = label(1:Nt,:);             % 训练样本对应的标签   the labels for the training samples
[tx,para1,para2]  =  data_normalize(tx0,scaler) ; % 训练集归一化 ，para1,para2用于归一化测试集   normalize the samples
randbatch_temp = rands{Nt/10};   %  the minibatch are created according this random matrix 

% 22 testing datasets are created
for i = 1:8
    vx{i}   = data(Nt+1+(i-1)*Nv:Nt+i*Nv,:);                        % 验证样本   the testing sample 
    [vx{i}, mu, sigma] = data_normalize(vx{i},scaler,para1,para2);  % 测试集归一化，使用训练集的参数   the testing samples are normalized 
    vx{i}   = vx{i}(1:Nv,:);                                        
    vy{i}   = label(Nt+1+(i-1)*Nv:Nt+i*Nv,:);                       % 验证样本对应的标签
end
Nt = 2100;
for i = 9:15
    vx{i}   = data(Nt+1+(i-1-8)*Nv:Nt+(i-8)*Nv,:);                  % 验证样本
    [vx{i}, mu, sigma] = data_normalize(vx{i},scaler,para1,para2);  % 测试集归一化，使用训练集的参数
    vx{i}   = vx{i}(1:Nv,:);
    vy{i}   = label(Nt+1+(i-1-8)*Nv:Nt+(i-8)*Nv,:);                 % 验证样本对应的标签
end
Nt = 2200;
for i = 16:22
    vx{i}   = data(Nt+1+(i-1-15)*Nv:Nt+(i-15)*Nv,:);                % 验证样本
    [vx{i}, mu, sigma] = data_normalize(vx{i},scaler,para1,para2);  % 测试集归一化，使用训练集的参数
    vx{i}   = vx{i}(1:Nv,:);
    vy{i}   = label(Nt+1+(i-1-15)*Nv:Nt+(i-15)*Nv,:);               % 验证样本对应的标签
end



%% 网络参数设置 the parameters of the networks
num_innode  = size(data,2);      % 输入层的节点数就是样本的维数     
num_outnode = size(label,2);     % 输出层的节点数就是样本的种类数，也是二值化后的标签的维数
netsize = [num_innode  46 42 38 34 30 26 22 18 14 10 7 num_outnode];         % 用于齿轮故障数据    50 32 22 12 5 ；  25 18 12 5   ； 11 9 7 5  ； 12 9 7 5 


hyperpara.k_PReLU = 0.2;            % hyperparameter for PReLU only
hyperpara.thp = 0                   %th_god(i_th) % th_god(i_th);
hyperpara.thn = -1.5                %-th_god(i_th) %-th_god(i_th);
% the below params for pretraining
switch nettype                                            % 根据在宏观设置中选择的网络类型进行选择  the nettype are set in godfinger
    case 'DNN'  % DNN design and pretraining
        sae.opts.netsize   = netsize;                     % DNN的网络结构   the netsize of the BDN   
        sae.opts.randbatch = randbatch_temp;              % 用于暂时控制批次随机    the minibatch are created according this random matrix
        sae.opts.numepochs = 20;                          % RBM预训练循环次数     the number of epoches for pretraining 
        sae.opts.batchsize = 25;                          % 设置RBM预训练的batch的大小  the bachsize for pretraining 
        sae.opts.lr        = prelr;                       % 设置RBM预训练的初始学习速率   the lr for pretraining 
        sae.opts.lr_adjust = prelr_adjust;                % 设置RBM预训练的学习速率调整因子   the lr are adjusted by this paramete during pretraining process
        sae.opts.momentum  = premom;                      % 设置RBM预训练的动量项    the momentum during pretraining process
        sae.opts.output_function      = acfun;            % 设置sae网络顶层分类器  the classifier function for the pretraining of DNN
        sae.opts.activation_function  = acfun;            % the activation function for the pretraining of DNN
        
        sae  = saesetup(sae,rand_initial,hyperpara);      % 调用子函数建立sae网络  SAEs are built
        sae  = saetrain(sae, tx);                         % 预训练sae   SAEs are pretrained
        nn   = nnassemble_sae(sae);                       % 调用子函数，将预训练得到的权值与偏置值重新组装成一个nn网络    the trained SAEs are taken to bulit a complete DNN     
end
% the blow params are designed for BP process
nn.opts.netsize              = netsize;   
nn.opts.randbatch            = randbatch_temp;            % 用于暂时控制批次随机     the minibatch are created according this random matrix
nn.opts.lr                   = lr;                        % 设置nn网络的初始学习速率  the lr for BP training 
nn.opts.lr_adjust            = lr_adjust;                 % 设置nn网络的学习速率调整因子  the lr are adjusted by this paramete during BP process
nn.opts.momentum             = mom;                       % 设置nn网络的动量项   the momentum during BP process
nn.opts.output_function      = 'softmax';                 % 设置nn网络顶层分类器   the classifier function for the pretraining of DNN
nn.opts.activation_function  = acfun;                     % 设置nn网络的传递函数  the activation function for the pretraining of DNN
nn.opts.numepochs            = 200;                         % nn训练循环次数   the number of epoches for pretraining 
nn.opts.batchsize            = 25;                        % nn的batch大小   the bachsize for pretraining 
nn.opts.plot                 = 0;                         % 是否画图，是：1，否：0    do you want to get the curves of training accuracy, etc. 


nn.opts.k_god  = 0.15; 
nn.opts.th_god = 4; 
nn = nntrain(nn, tx, ty, nn.opts, vx, vy);     % 调用子函数训练nn     nntrain are called for BP process



%% print and save
% the title and path for the plot and result excel
address_1th = strcat(filename,'\');
switch flag            % 根据不同的flag确定保存的地址名
    case 'acfuns'      % nnmain作为子函数，比较不同的传递函数
        subtitle    = strcat(flag , ' ' , ' (' , nn.opts.activation_function , ')' );     % 图形副标题设置，用于说明该图是通过什么传递函数，th&k得到的
        address_2th = strcat(address_1th, subtitle );          % 相关图片保存地址
    case 'th&k'        % nnmain作为子函数，比较不同的th&k
        subtitle    = strcat(flag , ' ' ,  ' (' , num2str(th_god(i_th)*100) , '&' , num2str(k_god(j_th)*100) , ' )' ); 
        %subtitle    = strcat(flag , ' ' ,  ' (' , num2str(k_god(i_k)*100) , '&' , num2str(1) , ' )' );   % 只针对ReLUs
        address_2th = strcat(address_1th, subtitle );          % 相关图片保存地址
        %address_2th = strcat(address_1th, subtitle ,',k=',num2str(nn.opts.k_god));

end
mkdir(address_2th);       % 根据提供的地址等建立一个新目录   create a file to store the result




%   the curves are created and saved
% {
fprintf('ploting now .... \n');
nnfigure(nn.tracking.accuracy_train , address_2th , strcat('training accuracy rate ~~ ',subtitle ) , 'epoch' , 'Classification rate');      % 训练正确率曲线
nnfigure(nn.tracking.loss_train     , address_2th , strcat('training error ~~ ',subtitle ) ,  'epoch' , 'Error');                           % 训练误差曲线
nnfigure(nn.tracking.lr             , address_2th , strcat('learningrate ~~ ',subtitle )   , 'epoch' , 'LearningRate');                     % 反向调节学习速率曲线
if strcmp(acfun,'hreltanh_opt')    % the hyperparams thp and thn are ploted for hreltanh_opt only
    for i = 1: length(nn.net)-2
        nnfigure([nn.tracking.thp(i,:);nn.tracking.thn(i,:)]       , address_2th , strcat('th',num2str(i) ,'~~ ',subtitle )   , 'epoch' , strcat('th',num2str(i)));     
    end
end
for i = 1:4   %length(nn.tracking.accuracy_val)     % 依次检索每个测试集的结果
    nnfigure(nn.tracking.accuracy_val{i}   , address_2th , strcat('validation accuracy rate (set', num2str(i), ') ~~ ',subtitle ) , 'epoch' , 'Classification rate');    % 验证正确率曲线
    nnfigure(nn.tracking.loss_val{i}       , address_2th , strcat('validation error (set', num2str(i), ') ~~ ',subtitle )  , 'epoch' , 'Error');                         % 验证误差曲线
end
%}

% 参数输出  the training and teating accuracies and losses are listed and saved
accuracy_loss_val = [];
for i = 1:length(nn.tracking.accuracy_val)     % 依次检索每个测试集的结果
    accuracy_loss_val = [accuracy_loss_val ;  nn.tracking.accuracy_val{i}(end)];    % 验证收敛正确率与误差];
end
nn.tracking.performance = [ 
                            nn.tracking.accuracy_train(end) ;          % 训练收敛正确率    
                            nn.tracking.loss_train(end)  ]                      
% 根据flag确定如何保存这些数据   
switch flag
    case 'acfuns'
        xlswrite(strcat(address_1th,'\acc&loss_result.xls'),nn.tracking.performance', flag ,strcat('C',num2str(i_acfun+1)));   % 将所有结果按行依次保存。（括号中分别是保存的地址，需要保存的矩阵，保存到的表单，位置）
        xlswrite(strcat(address_1th,'\acc&loss_result.xls'),accuracy_loss_val', flag ,strcat('H',num2str(i_acfun+1)));   % 将所有结果按行依次保存。（括号中分别是保存的地址，需要保存的矩阵，保存到的表单，位置）
end

fprintf('《《《《《《《《《《《《《《 sub over 》》》》》》》》》》》》》》》》》\n');

