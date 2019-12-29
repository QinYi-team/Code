% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数----将rbms与classifier组合成NN网络，以便进行反向传播
% the rbms and the classifier of DBN are combined and assembled as a NN for
% backpropagation process
function nn = nnassemble_dbn(dbn, hyperpara)           % dbn中包含完成RBM预训练后所有的权值、偏置值等

numlayer = numel(dbn.opts.netsize);     % 网络总层数

nn.net{1}.w = [0];        % 网络的第一层的相关的权值偏置值都是0，此处构想的每一层包括本层的节点，偏置值以及本层向前一层伸出的权值连线，实际上第一层只是一个装饰，没有作用，在实际中，第一层对样本不会进行任何处理
nn.net{1}.c = [0];

for i=1:length(dbn.rbm)               % 依次检索每个RBM
    nn.net{i+1}.w = dbn.rbm{i}.w;     % 依次将完成预训练的RBM的权值与偏置值提取出来组装成nn的隐含层
    nn.net{i+1}.c = dbn.rbm{i}.c;
    
    nn.net{i+1}.k_PReLU = hyperpara.k_PReLU ;  %*ones(1,dim(2)) ;
    nn.net{i+1}.thp = hyperpara.thp ;          %*ones(1,dim(2)) ;
    nn.net{i+1}.thn = hyperpara.thn;           %*ones(1,dim(2)) ;
end

nn.net{numlayer}.w = dbn.classifier.w;    % 输出层组装
nn.net{numlayer}.c = dbn.classifier.c;