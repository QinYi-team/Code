from __future__ import print_function          # 调用Python未来的输出函数print
import argparse                                # 参数解析模块
import torch

random_seed = 2
device = torch.device(["cuda","cpu"][0]) 
torch.manual_seed(random_seed)            # 为CPU设置随机数种子，使得模型最终结果是确定的
if device=="cuda":   # 如果使用cuda
    torch.cuda.manual_seed(random_seed)   # 为GPU设置随机数种子  

import torch.nn as nn
import torch.nn.functional as F                
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio 
import numpy as np
import random
import math
from sklearn import preprocessing
import datetime
from net_utils import  *
#from net_utils_48x64 import  NetConv2d
from data_loader_utils import MyDataset
from estimator_utils import plot_curve,plot_distribution_tsne
from final_test import final_test, data_loader_4final_test,save_excel_4final_test
from train_utils import train, test


CFG = dict(
          train_or_test = ['0-train',
                           '1-retrain_in_target_only','2-retrain_in_source_only','3-retrain_in_source&target',
                           '4-test','5-test_4trans','6-test_group'][0][2:]            # 训练还是测试的选择
          
        , dataset_choose = ['PlanetGearRaw4Ches_3072_Ch1',
                            'Case_Rawsignal_DEfault_12k_DE_&_normal_12k_DE_3072_load0'
                            ][0]   

        , dataset_choose_target =['PlanetGearRaw4Ches_3072_Ch1',
                                  'Case_Rawsignal_DEfault_12k_DE_&_normal_12k_DE_3072_load0'                                  
                                  ][0]  
        , random_seed = random_seed
        , device = device                 # 判断设备是不是GPU，torch.device代表将torch.Tensor分配到的设备的对象，包含cpu和cuda
        
        , batch_size = 64
        , test_batch_size = 64
        , num_epoch = 20
        , lr = 0.001
        , momentum = 0.2
        , gamma = 0.95
        
        , print_gap = 100 
        , epoch_gap = 30
        , save_gap = 50
        , save_model = True
        
        , NO_train = [[0,2000]][0]
        , NO_test  = [[2000,3000]][0]
        , num_class = 5        # 输出层神经元数目
        )

def main_(model_main):
#def main(CFG): 
    model = model_main.to(CFG['device'])     
      
# 模型训练、补训、测试 #################################################################        
    if CFG['train_or_test'] =='train': 
        #optimizer = optim.Adadelta(model.parameters(), lr=CFG['lr'])          # 求解器
        optimizer = optim.SGD(model.parameters(), lr=CFG['lr'], momentum=CFG['momentum'],)   
        scheduler = StepLR(optimizer, step_size=2, gamma=CFG['gamma'])        # 学习速率衰减方式
        result_trace = np.zeros([1,7])
        loss_trace = np.zeros([1,12])    # 逐个记录每个epoch的loss和acc
        acc_trace = np.zeros([1,12])      
        for epoch in range(0,20):                                # 遍历每个epoch
            start_time = datetime.datetime.now()          # 训练开始时间
            train_loss_epoch_i, train_acc_epoch_i = train(CFG, model, train_loader, optimizer, epoch)   # 
            end_time = datetime.datetime.now();   time_cost = (end_time - start_time).seconds ;   print('耗时:',time_cost)             # 训练耗时
            test_loss_epoch_i, test_acc_epoch_i, f4t_and_label = test(CFG, model, test_loader, is_print=True)

            result_epoch_i = [epoch, train_acc_epoch_i[0], train_loss_epoch_i[0], test_acc_epoch_i[0],test_loss_epoch_i[0], scheduler.get_lr()[0],time_cost]
            
            result_trace = np.vstack([result_trace,  np.array(result_epoch_i).reshape(1,len(result_epoch_i))]) 
            loss_trace = np.vstack([loss_trace,  np.array([train_loss_epoch_i + test_loss_epoch_i]).reshape(1,12)]) 
            acc_trace = np.vstack([acc_trace,  np.array([train_acc_epoch_i + test_acc_epoch_i]).reshape(1,12)]) 
            if epoch>0:
                scheduler.step()     
        if CFG['save_model'] :  
            pt_name = '[STI]_' + model.name+ '_'+CFG['dataset_choose'] +'_epoch'+str(0)+'to'+str(epoch) +'_sample'+str(CFG['NO_train'][0])+'to'+str(CFG['NO_train'][1])     # 文件主命名,  STI表示Source Trained In
        #    plot_curve(result_trace[:,0],[result_trace[:,1],result_trace[:,3]],'结果图/'+pt_name+'_ACC.png',  xlabel='Epoch',ylabel='ACC',title='ACC',legend=['Training_Accuracy','Testing_Accuracy'])
            pt = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'CFG':CFG,
                  'model_name':model.name, 'result_trace':result_trace, 'loss_trace':loss_trace, 'acc_trace':acc_trace }
            torch.save(pt,    pt_name  + '_' + device.type + '_rand' + str(CFG['random_seed']) + '.pt')



model_pool = [NetConv1d_MSTVM(CFG)]    # 选择模型


srcdata_pool = ['PlanetGearRaw4Ches_3072_Ch1',
                'PlanetGearRaw4Ches_3072_Ch3',
                'PlanetGearRaw4Ches_3072_Ch4',]
tardata_pool = ['PlanetGearRaw4Ches_3072_Ch4',
                'PlanetGearRaw4Ches_3072_Ch3',
                'PlanetGearRaw4Ches_3072_Ch1',]

for id in range(3):
    CFG['dataset_choose'] = srcdata_pool[id]
    CFG['dataset_choose_target'] = tardata_pool[id]
    # 数据集加载 #################################################################   torch.manual_seed()   
    train_data = MyDataset(CFG['dataset_choose'],CFG['NO_train'][0],CFG['NO_train'][1])         # 训练数据
    test_data  = MyDataset(CFG['dataset_choose'],CFG['NO_test'][0],CFG['NO_test'][1])           # 测试数据 
    train_loader = DataLoader(dataset=train_data,  batch_size=CFG['batch_size'],  shuffle=False)
    test_loader  = DataLoader(dataset=test_data,   batch_size=CFG['test_batch_size'],  shuffle=False)
    
    if CFG['train_or_test'] in ['retrain_in_source&target','test','test_4trans']:     # 只有源域和目标域都需要训练的时候才制作目标域样本
        train_data_target = MyDataset(CFG['dataset_choose_target'],CFG['NO_train'][0],CFG['NO_train'][1])     # target表示目标域，源域没有加source
        test_data_target  = MyDataset(CFG['dataset_choose_target'],CFG['NO_test'][0],CFG['NO_test'][1])
        train_loader_target = DataLoader(dataset=train_data_target,  batch_size=CFG['batch_size'],  shuffle=False)
        test_loader_target  = DataLoader(dataset=test_data_target,   batch_size=CFG['test_batch_size'],  shuffle=False)

    # ================================================
    for im in [0]: #range(11):
        model_main = model_pool[im]    # 选择模型
        main_(model_main)




    
