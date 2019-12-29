% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 子函数——DBN中的每层RBM预训练的总控制函数
% RBM (the basic block is pretrained as following)
function dbn = dbntrain(dbn, x)
n = numel(dbn.rbm);    % RBM的个数

% 第一个RBM（之所以将RBM1独立出来是因为RBM1的输入值是原始数据，而之后的RBM的输入值是由前一个RBM计算来的）
disp(['%%%%%%%%%%%%%%% rbm1 %%%%%%%%%%%%%%%%']);     % 输出一个标记，以便观察进度
dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, dbn.opts);      % 训练RBM

% 后续的RBM
for i = 2 : n      % 依次检索每个RBM
    disp(['%%%%%%%%%%%%%%% rbm',num2str(i),' %%%%%%%%%%%%%%%%']);    % 输出一个标记，以便观察进度
    x = rbmup(dbn.rbm{i - 1}, x);                                    % 根据前一层RBM预训练得到的权值与偏置值计算本层RBM的输入值
    dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, dbn.opts);                  % 训练RBM
end


