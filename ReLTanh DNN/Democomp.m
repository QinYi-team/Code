% =========================================================================
%                          Written by Xin Wang and Yi Qin
% =========================================================================

%% 主函数——可用来操控nnmain，以使其进行循环计算
% main can be used to call the nnmain

clear all
close all

% add the sub path
addpath(genpath('activation_functions_utils'))
addpath(genpath('datasets'))
addpath(genpath('data_utils'))
addpath(genpath('net_utils'))
addpath(genpath('results'))
%% Compare various activation functions
acfuns = { 'hreltanh_opt', 'tanh','ReLU','LReLU', 'softplus', 'ELU', 'swish', 'hexpo'};    % various acfuns are offered for comparison   hreltanh_opt is tthe "ReLTanh" proposed in the paper
nettype = 'DNN'   ;   % three kinds of net are offered, BPNN and DNN 
for i_rep = 1:4       % repeatedly run the models with different initialization, i_rep can be increased if you want to get a complete evaluation.
    filename = strcat('results\','ReLTanh_4planetGear_Fault_',nettype,'_repeat',num2str(i_rep));    % create a file to save the result
    mkdir(filename);                                          % create a file to save the result
    copyfile('datasets/acc&loss_result.xls',filename);        % copy the excel to the result file
    for i_acfun = 1:length(acfuns)   % try the activation functions one by one
        nn = nnmain(acfuns{i_acfun}, i_rep , 'acfuns', i_acfun, nettype,filename);          % the nnmain are called   
    end
end


fprintf('《《《《《《《《《《《《《《 all over 》》》》》》》》》》》》》》》》》\n');









