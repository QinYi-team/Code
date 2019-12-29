% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================
%% 子函数----数据归一化处理
% the samples are norimalized below and three kind of scaler are offered
function [data_ed,para1,para2] = data_normalize(data,scaler,para1,para2)

% minmax 方式
if strcmp(scaler,'minmax')
    if nargin == 2              % 如果只传入了一个参数，那就是对训练集进行归一化，此时需要计算datamin与datamax
        para1 = min(data);      % 提取每一列的最小值 
        para2 = max(data);      % 提取每一列的最大值
    end
    data_ed = bsxfun(@rdivide,bsxfun(@minus,data,para1),(para2-para1));      % max-min归一化； %C=bsxfun(FUNC,A,B)按照FUNC函数句柄对矩阵A和B进行元素一对一的运算，
                                                         %rdivide矩阵右除，minus矩阵相减  

% z-score方式                                                      
elseif strcmp(scaler,'maxabs')
    if nargin == 2                  % 如果只传入了一个参数，那就是对训练集进行归一化，此时需要计算datamin与datamax
        para1 = 0;	             % 均值
        para2 = max(max(abs(data)),eps);    % std(x)按列求出每一列的标准差;   eps(x)求出abs（x）到最近一个浮点数的证书距离,  默认：eps=eps（1）=2.2204e-16    ； 这里的max就是找出eps矩阵和实体店矩阵中的最大值，目的是将std中的0替换为eps（1），因为0不能再充当分母
    end    
	data_ed=bsxfun(@minus,data,para1);        % bsxfun（fun,A,B）对两个矩阵A和B之间的每一个元素进行指定的计算（函数fun指定）；   minus（A,B）=  A -B
	data_ed=bsxfun(@rdivide,data_ed,para2);   % rdivide(A,B) = A ./ B   实现点除功能    
                                                             
                                                         
                                                         
% z-score方式                                                      
elseif strcmp(scaler,'z-score')
    if nargin == 2                  % 如果只传入了一个参数，那就是对训练集进行归一化，此时需要计算datamin与datamax
        para1 = mean(data);	        % 均值
        para2 = max(std(data),eps);    % std(x)按列求出每一列的标准差;   eps(x)求出abs（x）到最近一个浮点数的证书距离,  默认：eps=eps（1）=2.2204e-16    ； 这里的max就是找出eps矩阵和实体店矩阵中的最大值，目的是将std中的0替换为eps（1），因为0不能再充当分母
    end    
	data_ed=bsxfun(@minus,data,para1);        % bsxfun（fun,A,B）对两个矩阵A和B之间的每一个元素进行指定的计算（函数fun指定）；   minus（A,B）=  A -B
	data_ed=bsxfun(@rdivide,data_ed,para2);   % rdivide(A,B) = A ./ B   实现点除功能

    

    
end                                      
                                                     