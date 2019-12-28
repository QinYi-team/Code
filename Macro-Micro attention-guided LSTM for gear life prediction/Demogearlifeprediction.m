% =========================================================================
%                          Written by Yi Qin and Sheng Xiang
% =========================================================================
clear all;
close all;
tic%time
clc;
load traindata
load testdata
%input the cell number of the input layer ,learning rate,traindata,testdata ;
%in this example,380 healthy characteristic points are known, we suggest that input layer and learning rate can be respectively 35 and 0.05
%For other data sets with different known points, input layer and learning rate shoud be properly set as other optimal values
life = MMA_LSTM( 35,0.05,traindata,testdata );
toc