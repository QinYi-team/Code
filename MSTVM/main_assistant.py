import torch
import torch.nn as nn
import os
import math
import data_loader
import models
from config import CFG
import utils
from data_loader_utils import MyDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim.lr_scheduler import StepLR
from estimator_utils import plot_curve,plot_distribution_tsne
from train_utils import train, test
from final_test import final_test, data_loader_4final_test,save_excel_4final_test
import xlwt
import datetime

DEVICE = CFG['DEVICE']



# ===================================================================
# 源域与目标域的训练与测试集样本
def loaders_creator(CFG,nvs,nve):        # nvs：验证样本起始位置，  nve：验证样本终止位置
    batch_size = CFG['batch_size']
    batch_size_test = CFG['batch_size_test']
    train_loader_src = DataLoader(dataset=MyDataset(CFG['src_name'],0,nvs),  batch_size=batch_size,  shuffle=False)   # DataLoader默认不会抛弃最后一个不完整的batch
    test_loader_src  = DataLoader(dataset=MyDataset(CFG['src_name'],nvs,nve),   batch_size=batch_size_test,  shuffle=False)   
    train_loader_tar = DataLoader(dataset=MyDataset(CFG['tar_name'],0,nvs),  batch_size=batch_size,  shuffle=False)
    test_loader_tar  = DataLoader(dataset=MyDataset(CFG['tar_name'],nvs,nve),   batch_size=batch_size_test,  shuffle=False)
    loaders = [train_loader_src,test_loader_src, train_loader_tar, test_loader_tar]
    return loaders





# ===================================================================
# 模型参数包
def spparams_creator(model,CFG):
    '''
    params = [  {'params': model.FE_net.parameters()},                 #  model.base_network.parameters()
                {'params': model.Mid_net.parameters(), 'lr':  CFG['lr']},
                {'params': model.Clf_net.parameters(), 'lr': CFG['lr']},
                {'params': model.DomClf_net.parameters(), 'lr': CFG['lr']} ]   
    '''
    if ('RevGrad' in CFG['tranmodel']):  # 对于RevGrad组建参数包
        params = [  {'params': model.model.parameters()} ]     # FE与TC的参数包
        params_domclf = [ {'params': model.DomClf_net.parameters(), 'lr': 1*CFG['lr']} ]      # 域判别器的参数包
        params = [params, params_domclf]
    else:     # 对于DDC三大模型的参数包
        params = [  {'params': model.model.parameters()}, ]                # FE与TC的参数包

    return params
    
    


