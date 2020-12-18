import torch
import os
import math
from collections import Counter
import data_loader
import models
import utils
from data_loader_utils import MyDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from estimator_utils import plot_curve,plot_distribution_tsne

import mmd
from assistant_utils import feamap_segmenter,voter_multi_classifier,ReverseLayerF,RevGrad_assistant,loss_adapt_fun
from test_utils import test
import xlwt  # 负责写excel
import xlrd



# ===================================================================
# 生成平行测试集
def data_loader_4final_test(nts,nte,nt,nt_gap,CFG): # nts：测试样本起始位置，  nte：测试样本终止位置    nt：每个测试集的样本数    nt_gap：测试集取样间隔
    # 组装测试集
    nte = nte
    bt_test = CFG['batch_size_test']
    test_loaders_src, test_loaders_tar = [],[]    # 目标域测试集样本的容器    
    num_cycle = int(nt / nt_gap - 1)              # 循环数，每个循环内部会提取出多个测试集
    
    for i_cycle in range(num_cycle):         # 遍历所有循环
        nts = nts + i_cycle * nt_gap         # 测试集起始位置挪动
        for i in range(int((nte-nts)/nt)):   # 每个循环内部的测试集数
            test_loader_tar = DataLoader(dataset=MyDataset(CFG['tar_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
            test_loaders_tar.append(test_loader_tar)
            test_loader_src = DataLoader(dataset=MyDataset(CFG['src_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
            test_loaders_src.append(test_loader_src)
    return test_loaders_src, test_loaders_tar            


# ===================================================================
# 对每个测试子集逐个测试
def final_test(model,CFG,final_test_loaders):       
    test_loaders_src, test_loaders_tar = final_test_loaders   # 测试集，源域与目标域的
    # 开始测试  
    acc_test_srcs, acc_test_tars = [], []    # 正确率容器
    for i in range(len(test_loaders_tar)):   # 遍历所有测试集
        #test_loader_src, test_loader_tar = test_loaders_src[i], test_loaders_tar[i]    # 一个有s，一个没有
        acc_collection_test, loss_collection_test,_ = test(model, test_loaders_src[i], test_loaders_tar[i] , 0, CFG)      # 依次对每个测试集进行测试   
        acc_test_srcs.append(acc_collection_test[0])    # 将第0个和第6个测试集的正确率作为最终投票结果
        acc_test_tars.append(acc_collection_test[6])
    return acc_test_srcs, acc_test_tars        



# ===================================================================
# 将平行测试集的结果写入到表格中
def save_excel_4final_test(acc_4excel_src,acc_4excel_tar,   excel,sheet_src,sheet_tar,   im,CFG):    
    header_col = 2       # 空表头，用于填写新信息
    header_row = 2
    # 表单1，保存目标域的测试集的测试结果  
    sheet_tar.write(im+header_row,1,CFG['tranmodel'])
    num_test = len(acc_4excel_tar)     # 测试集数目
    for j in range (num_test):    # 依次遍历所有测试集
        sheet_tar.write(im+header_row,j+ header_col,acc_4excel_tar[j])   # 将正确率数值按要求写入excel中
    
    # 表单2，保存源域的测试集的测试结果    
    sheet_src.write(im+header_row,1,CFG['tranmodel'])
    num_test = len(acc_4excel_src)      # 测试集数目
    for j in range (num_test):
        sheet_src.write(im+header_row,j+ header_col,acc_4excel_src[j])        
    













