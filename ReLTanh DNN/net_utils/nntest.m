% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
% for validation process 
function [er, bad] = nntest(nn, x, y)    % 调用格式 [er_train, dummy]   = nntest(nn, train_x, train_y);
    labels = nnpredict(nn, x);      % 得到训练数据的输出结果
    [dummy, expected] = max(y,[],2);    % 得到期望输出结果
    bad = find(labels ~= expected);     % 得到判别失败的结果
    er = numel(bad) / size(x, 1);     % 得到错误率
end
