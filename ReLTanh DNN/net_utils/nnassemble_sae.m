% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数----将所有ae组合成NN网络，以便进行反向传播
% the SAEs and the classifier of DNN are combined and assembled as a NN for
% backpropagation process
function nn = nnassemble_sae(sae)           % dbn中包含完成RBM预训练后所有的权值、偏置值等

numlayer = numel(sae.opts.netsize);     % 网络总层数

nn.net{1}.w = [0];        % 网络的第一层的相关的权值偏置值都是0，此处构想的每一层包括本层的节点，偏置值以及本层向前一层伸出的权值连线，实际上第一层只是一个装饰，没有作用，在实际中，第一层对样本不会进行任何处理
nn.net{1}.c = [0];

for i=1:length(sae.ae)               % 依次检索每个RBM
    nn.net{i+1}.w = sae.ae{i}.net{2}.w;     % 依次将完成预训练的RBM的权值与偏置值提取出来组装成nn的隐含层
    nn.net{i+1}.c = sae.ae{i}.net{2}.c;
    nn.net{i+1}.k_PReLU = sae.ae{i}.net{2}.k_PReLU;
    nn.net{i+1}.thp = sae.ae{i}.net{2}.thp;
    nn.net{i+1}.thn = sae.ae{i}.net{2}.thn;
end
    nn.net{numlayer}.w = sae.classifier.w;    % 输出层组装
    nn.net{numlayer}.c = sae.classifier.c;