% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 子函数——DBN网络结构建立
% the DBN are built as following
function dbn = dbnsetup(dbn,rand_initial)
netsize = dbn.opts.netsize;    % 网络的层数与每层的节点数
num_rbm = length(netsize)-2;   % RBM的个数
dbn.rbm = cell(num_rbm,1);     % 建立存储每个RBM相关数据的元胞数组

% 各层RBM
for i= 1:num_rbm    % 依次检索每个RBM
    %{
    dim = [netsize(i),netsize(i+1)];     % 当前RBM的维数
    r   = 4*sqrt(6./(dim(1)+dim(2)));    % 一个经验参数，广泛被用来初始化设计RBM的权值
    
    dbn.rbm{i}.w  = 2*r*rand(dim(1),dim(2))-r;      % 当前RBN的权值矩阵
    dbn.rbm{i}.b  = zeros(1,dim(1));                % 当前RBM的两个偏置值向量
    dbn.rbm{i}.c  = zeros(1,dim(2));
    dbn.rbm{i}.vw = zeros(size(dbn.rbm{i}.w));      % 用来存储权值、偏置值的更新值
    dbn.rbm{i}.vb = zeros(size(dbn.rbm{i}.b));
    dbn.rbm{i}.vc = zeros(size(dbn.rbm{i}.c));
    %}
    
    dim         = [netsize(i),netsize(i+1)];      % 网络结构
    temp        = sqrt( 3 / dim(1));
    %temp        = sqrt( 6 /(dim(1) + dim(2)));
    dbn.rbm{i}.w = 2*(rand_initial{i}(1:dim(1),1:dim(2))- 0.5) * temp;  % ;    
    dbn.rbm{i}.b  = zeros(1,dim(1));                % 当前RBM的两个偏置值向量
    dbn.rbm{i}.c  = zeros(1,dim(2));
    dbn.rbm{i}.vw = zeros(size(dbn.rbm{i}.w));      % 用来存储权值、偏置值的更新值
    dbn.rbm{i}.vb = zeros(size(dbn.rbm{i}.b));
    dbn.rbm{i}.vc = zeros(size(dbn.rbm{i}.c));    
    
end

% 顶层classifier
%{
r=4*sqrt(6./(netsize(end-1)+netsize(end)));                     % 顶层分类器的r经验参数
dbn.classifier.w = 2*r*rand(netsize(end-1),netsize(end))-r;     % 顶层分类器的权值矩阵
dbn.classifier.c = zeros(1,netsize(end));                       % 顶层分类器的偏置值向量
%}
dim         = [netsize(end-1),netsize(end)];      % 网络结构
temp        = sqrt( 3 / dim(1));
%temp        = sqrt( 6 /(dim(1) + dim(2)));
dbn.classifier.w = 2*(rand_initial{end}(1:dim(1),1:dim(2))- 0.5) * temp;  % ;
dbn.classifier.c = zeros(1,dim(2));                % 偏置值初始化为0


