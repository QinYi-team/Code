% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 子函数——SAE中的每层AE预训练的总控制函数
function sae = saetrain(sae, x)

for i = 1 : numel(sae.ae);       % 依次检索每个编码器
    disp(['%%%%%%%%%%%%%%% ae',num2str(i),' %%%%%%%%%%%%%%%%']);    % 输出一个标记，以便观察进度
    vx{1} = x;    % 为了将就多个测试结果而设置
    sae.ae{i} = nntrain(sae.ae{i}, x, x, sae.ae{i}.opts, vx, vx);           % 调用nntrain训练ae，其标准输出值就是x
    t = nnff(sae.ae{i}, x, x);                                      % 用上一个ae的隐含层输出值计算下一个ae的输出值
    x = t.net{2}.out;                                               % 用上一个ae的隐含层输出值计算下一个ae的输出值
    %remove bias term
    %x = x(:,2:end);
end

 