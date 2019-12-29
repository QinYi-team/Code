% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数----标签二值化处理
% the labels are binarize
function label_ed=data_binarize(label)
label = label + 1;     % 对十进制的标签进行加一，因为对于minist数据库的标签是0-9的
num_sample=length(label);     % 标签的个数，也就是样本的个数； length（）计算矩阵中维数最大的维
num_label=length(unique(label));      % 给出标签的种类，也就是样本的种类数；unique(label)剔除重复的，对每一种只提取一个出来；length（）进一步计算出种类的数目
label_ed=zeros(num_sample,num_label);    % 根据样本种类数与总数设计二值化后的标签存储的零矩阵

[value,~] = max(label,[],2);    % 确定每一行标签的最大值，作为置入1的位置
for i=1:num_sample     % 依次检索每个样本标签
    label_ed(i,value(i)) = 1;    % 将零矩阵每行对应位置的0换为1
end

            
            