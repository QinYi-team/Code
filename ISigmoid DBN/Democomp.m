% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 主函数——可用来操控nnmain，以使其进行循环计算
% godfinger can be used to call the nnmain

clc
clear all
close all

% add the sub path
addpath(genpath('activation_functions_utils'))
addpath(genpath('datasets'))
addpath(genpath('data_utils'))
addpath(genpath('net_utils'))
addpath(genpath('randfiles'))
addpath(genpath('results'))
%% Compare various activation functions
acfuns = { 'isigmoid','sigmoid','ReLU','LReLU', 'softplus', 'ELU', 'swish', 'hexpo'};    % various acfuns are offered for comparison
nettype = 'DBN';   % three kinds of net are offered, BPNN and DNN 

for i_rep = 1:3       % repeatedly run the models with different initialization, i_rep can be increased if you want to get a complete evaluation.
    filename = strcat('results\','ISigmoid_4planetGear_Fault_',nettype,'_repeat',num2str(i_rep));    % create a file to save the result
    mkdir(filename);                                          % create a file to save the result
    copyfile('datasets/acc&loss_result.xls',filename);        % copy the excel to the result file
    for i_acfun = 1:length(acfuns)   % try the activation functions one by one
        if strcmp(acfuns{i_acfun}, 'isigmoid')| strcmp(acfuns{i_acfun}, 'sigmoid')
            lr_god = 0.5             % Sigmoid family can perform better with greater lr
            nn = nnmain(acfuns{i_acfun}, i_rep , 'acfuns', i_acfun, nettype, lr_god, filename);          % the nnmain are called   
        else 
            lr_god = 0.005           % ReLU family can perform better with samller lr
            nn = nnmain(acfuns{i_acfun}, i_rep , 'acfuns', i_acfun, nettype, lr_god, filename);          % the nnmain are called                   
        end
    end

end


fprintf('《《《《《《《《《《《《《《 all over 》》》》》》》》》》》》》》》》》\n');





