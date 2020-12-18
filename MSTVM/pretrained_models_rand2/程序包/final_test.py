import torch
import os
import math
from collections import Counter
#import data_loader
#import models
#import utils
from data_loader_utils import MyDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from estimator_utils import plot_curve,plot_distribution_tsne
#from Coral import CORAL
#import mmd
#from assistant_utils import feamap_segmenter,voter_multi_classifier,ReverseLayerF,RevGrad_assistant, DCTLN_assistant,loss_adapt_fun
#from test_utils import test
import xlwt  # 负责写excel
import xlrd




def final_test(model,CFG,final_test_loaders,sppackt):       
    test_loaders_src, test_loaders_tar = final_test_loaders
    # 开始测试  
    acc_test_srcs, acc_test_tars = [], []
    for i in range(len(test_loaders_tar)):
        test_loader_src, test_loader_tar = test_loaders_src[i], test_loaders_tar[i]
        acc_collection_test, loss_collection_test = test(model, test_loader_src, test_loader_tar, 0, CFG,sppackt)        
        acc_test_srcs.append(acc_collection_test[0])
        acc_test_tars.append(acc_collection_test[7])
    return acc_test_srcs, acc_test_tars        




def save_excel_4final_test(acc_test_srcs, acc_test_tars,excel,sheet_src,sheet_tar,im,CFG):    
    header_col = 4       # 空表头，用于填写新信息
    header_row = 2
    # 表单1，保存目标域的测试集的测试结果  
    sheet_src.write(im+header_row,1,CFG['tranmodel'])
    num_test = len(acc_test_tars) #h为行数，l为列数
    for j in range (num_test):
        sheet_src.write(im+header_row,j+ header_col,acc_test_srcs[j])
    
    # 表单2，保存源域的测试集的测试结果    
    sheet_tar.write(im+header_row,1,CFG['tranmodel'])
    num_test = len(acc_test_tars) #h为行数，l为列数
    for j in range (num_test):
        sheet_tar.write(im+header_row,j+ header_col,acc_test_tars[j])        
    






def data_loader_4final_test(nts,nte_sub,nt,nt_gap,CFG):
    # 组装测试集
    nte = nte_sub
    bt_test = 64
    test_loaders_src, test_loaders_tar = [],[]    # 目标域测试集样本的容器    
    num_cycle = int(nt / nt_gap - 1)          # 循环数
    
    for i_cycle in range(num_cycle):
        for i in range(int((nte-nts)/nt)):
            test_loader_tar = DataLoader(dataset=MyDataset(CFG['tar_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
            test_loaders_tar.append(test_loader_tar)
            test_loader_src = DataLoader(dataset=MyDataset(CFG['src_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
            test_loaders_src.append(test_loader_src)
    return test_loaders_src, test_loaders_tar            
    '''
    nts = nts + nt_gap
    for i in range(int((nte-nts)/nt)):
        test_loader_tar = DataLoader(dataset=MyDataset(CFG['tar_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_tar.append(test_loader_tar)    
        test_loader_src = DataLoader(dataset=MyDataset(CFG['src_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_src.append(test_loader_src)
        
    nts = nts + nt_gap
    for i in range(int((nte-nts)/nt)):
        test_loader_tar = DataLoader(dataset=MyDataset(CFG['tar_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_tar.append(test_loader_tar)    
        test_loader_src = DataLoader(dataset=MyDataset(CFG['src_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_src.append(test_loader_src)
     
    nts = nts + nt_gap
    for i in range(int((nte-nts)/nt)):
        test_loader_tar = DataLoader(dataset=MyDataset(CFG['tar_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_tar.append(test_loader_tar)    
        test_loader_src = DataLoader(dataset=MyDataset(CFG['src_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_src.append(test_loader_src)

    nts = nts + nt_gap
    for i in range(int((nte-nts)/nt)):
        test_loader_tar = DataLoader(dataset=MyDataset(CFG['tar_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_tar.append(test_loader_tar)    
        test_loader_src = DataLoader(dataset=MyDataset(CFG['src_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_src.append(test_loader_src)        

    nts = nts + nt_gap
    for i in range(int((nte-nts)/nt)):
        test_loader_tar = DataLoader(dataset=MyDataset(CFG['tar_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_tar.append(test_loader_tar)    
        test_loader_src = DataLoader(dataset=MyDataset(CFG['src_name'],nts+i*nt,nts+(i+1)*nt),   batch_size=bt_test,  shuffle=False)
        test_loaders_src.append(test_loader_src)        
    '''    











