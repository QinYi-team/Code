% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数——跟踪记录每一次循环后的相关数据
% for tracking the states such as training accracies and losses during the
% training process
function tracking = nntracking(nn, tracking, train_x, train_y, val_x, val_y)   % 次数用的tarin_x值所有的训练

assert(nargin == 4 || nargin == 6, 'Wrong number of arguments');      % 如果传入的变量数不是4或者6 就报错，4 是没有验证集，6是包含验证集

% 训练数据集
nn = nnff(nn, train_x, train_y);         % 不分批，一次性计算
[~, output] = max(nn.net{end}.out,[],2);      % 找出输出层的最大值作为其判断的正确值
[~, label] = max(train_y,[],2);              % 二值化标签最大值作为其判断的正确值

row = find(output ~= label);             % 两者相比较，找出不相同的元素的坐标
tracking.fail_train              = [label(row),output(row)];                % 得到判别失败的结果
tracking.accuracy_train(end + 1) = 1 - (numel(row) / size(train_y, 1));     % 得到错误率
tracking.loss_train(end + 1)     = nn.loss;   % 训练误差

temp_thp = [];
temp_thn = [];
for i = 2: length(nn.net)-1
    temp_thp = [temp_thp;nn.net{i}.thp];  
    temp_thn = [temp_thn;nn.net{i}.thn];  
end
tracking.thp(:,end + 1)  = temp_thp;   
tracking.thn(:,end + 1)  = temp_thn;
    


% 验证数据集
if nargin == 6                            % 如果传入的变量数为6，就书名有验证集
    for i = 1: length(val_x)              % 依次检索每一组验证样本    
        nn = nnff(nn, val_x{i}, val_y{i});          % 一次性计算验证集
        [~, output] = max(nn.net{end}.out,[],2);   % 找出输出层的最大值作为其判断的正确值
        [~, label]  = max(val_y{i},[],2);             % 找出输出层的最大值作为其判断的正确值

        row = find(output ~= label);     
        tracking.fail_val{i}              = [label(row),output(row)];              % 得到判别失败的结果
        tracking.accuracy_val{i}(end + 1) = 1 - (numel(row) / size(val_y{i}, 1));     % 得到错误率
        tracking.loss_val{i}(end + 1)     = nn.loss;
    end

end



