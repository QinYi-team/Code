% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
% the predicted label of the model are gained below
function labels = nnpredict(nn, x)      % 调用格式：labels = nnpredict(nn, x);   % 将训练数据传递进来
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    [dummy, i] = max(nn.a{end},[],2);      % 找出输出层的最大值作为其判断的正确值
    labels = i;
end
