import torch
from config import CFG    # 参数包

model_index = [0,1]

DEVICE = CFG['DEVICE']                 #运行设备（GPU或者CPU）
rand_seed = 2                          # 随机数种子
rand_seed1 = 2
torch.manual_seed(rand_seed)           # 为CPU设置随机数种子，使得模型最终 结果是确定的
if DEVICE=="cuda":                     # 如果使用cuda
    torch.cuda.manual_seed(rand_seed)  # 为GPU设置随机数种子     
# 库函数导入
import torch.nn as nn
import os
import math
import numpy as np
from torch.optim.lr_scheduler import StepLR
import xlwt,xlrd
import datetime
from main_assistant import spparams_creator,loaders_creator
import data_loader
import models
import utils
from data_loader_utils import MyDataset
from torch.utils.data import Dataset, DataLoader
from estimator_utils import plot_curve,plot_distribution_tsne
from train_utils import train, test
from final_test import final_test, data_loader_4final_test,save_excel_4final_test


# 主函数
def main(loaders, final_test_loaders, excel_files, im):   
    excel,sheet_src,sheet_tar = excel_files      # 将excel的参数包解压出来
    
    # 参数包准备
    model = models.Transfer_Net(CFG).to(DEVICE)     # 创建model，并且将其拷贝到指定的device中
    params = spparams_creator(model,CFG)            # 生成用于模型训练的参数
    if ('RevGrad' in CFG['tranmodel']):     # 对于RevGrad模型，需要单独设计优化器与衰减器
        optimizer0 = torch.optim.SGD(params[0], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])  # 主干的参数
        optimizer1 = torch.optim.SGD(params[1], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])  # domclf的参数
        scheduler0 = StepLR(optimizer0, step_size=1, gamma=0.95)        # 学习速率衰减方式
        scheduler1 = StepLR(optimizer1, step_size=1, gamma=0.95)        # 学习速率衰减方式
        optimizer = [optimizer0,  optimizer1]   # 将两个优化器放在list中便于参数传递
        scheduler = [scheduler0,  scheduler1]   #
    else:       # 对于DDC优化器与迭代器分别只有1个
        optimizer = torch.optim.SGD(params, lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)        # 学习速率衰减方式

    
    # 开始训练
    result_trace = train(loaders, model, optimizer, scheduler,CFG)  # 用train_utils文件中的train函数进行训练

    # 绘制训练曲线       
    acc_train, acc_test = np.array(result_trace[0]),  np.array(result_trace[3])   # index=0是
    acc_train_src,acc_train_tar,    acc_test_src,acc_test_tar = acc_train[:,0],acc_train[:,6],  acc_test[:,0],acc_test[:,6]   # 第0列是源域最终投票正确率，第6列是目标域最终投票正确率
    x = [i  for i in range(len(acc_train_src))];   y = [acc_train_src, acc_train_tar,  acc_test_src,  acc_test_tar]    # x，y，用于画图
    plot_curve(x,y, path_name_curve+ '/'  + CFG['tranmodel']   + '_ACC_end.png',  xlabel='Epoch',ylabel='ACC',title=CFG['tranmodel']+'ACC',legend=['acc_train_src', 'acc_train_tar','acc_test_src',  'acc_test_tar'])   #调用自定义函数进行画图
    
    # 测试结果
    acc_train_val_src = [acc_train_src[-1],acc_test_src[-1]]    # 训练集和验证集的收敛正确率，源域和目标域
    acc_train_val_tar = [acc_train_tar[-1],acc_test_tar[-1]] 
    acc_test_srcs, acc_test_tars = final_test(model, CFG, final_test_loaders)      # 对多个平行测试集进行测试的正确率
    acc_4excel_src = acc_train_val_src + [0,0] + acc_test_srcs     # 将训练集验证集的正确率，[0，0](用于占位)，平行测试集的正确率  一起拼接成list，以便于写入excel
    acc_4excel_tar = acc_train_val_tar + [0,0] + acc_test_tars
    save_excel_4final_test(acc_4excel_src,acc_4excel_tar,   excel,sheet_src,sheet_tar,   im,CFG)
    excel.save(dir_now +'/测试结果.xls')



    # pt文件保存
    if ('RevGrad' in CFG['tranmodel']) :    # 对于RevGrad需要保存两个优化器与两个衰减器
        pt = {'model':model.state_dict(), 'optimizer0':optimizer[0].state_dict(), 'optimizer1':optimizer[1].state_dict(),'scheduler0':scheduler[0].state_dict(),'scheduler1':scheduler[1].state_dict(),
                     'CFG':CFG,  'result_trace':result_trace,  'test_result':[acc_test_srcs, acc_test_tars]}
    else:            # 对于DDC优化器与迭代器分别只有1个
        pt = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
             'CFG':CFG,  'result_trace':result_trace,  'test_result':[acc_test_srcs, acc_test_tars]}
    torch.save(pt,  dir_now + '/' + CFG['tranmodel']    +'_end.pt')
    

# ——————————————————————————————————————————————————————————————————————————
# ——————————————————————————————————————————————————————————————————————————
# 函数运行起点
srcdata_pool = ['PlanetGearRaw4Ches_3072_Ch1',    # 源域数据集
                'PlanetGearRaw4Ches_3072_Ch3']  
tardata_pool = ['PlanetGearRaw4Ches_3072_Ch3',    # 目标域数据集
                'PlanetGearRaw4Ches_3072_Ch1']
filename_pool = ['【测点迁移】Ch1toCh3_rand',      # 迁移任务
                 '【测点迁移】Ch3toCh1_rand']

# ============================================================
# 开启真实计算
for ie in range(2):   
    # 创建样本集loader ####################
    CFG['src_name'] = srcdata_pool[ie]
    CFG['tar_name'] = tardata_pool[ie]
    print('Src: %s  \nTar: %s' % (CFG['src_name'], CFG['tar_name']))
    loaders = loaders_creator(CFG,nvs=2000,nve=3000)   # nvs：验证样本起始位置，  nve：验证样本终止位置
    final_test_loaders = data_loader_4final_test(nts=2000,nte=5000,nt=300,nt_gap=100,CFG=CFG)
                   # nts：测试样本起始位置，  nte：测试样本终止位置    nt：每个测试集的样本数    nt_gap：测试集取样间隔
    
    # 判断存储路径是否存在 ####################
    dir_now = filename_pool[ie] + str(rand_seed)    # 文件保存地址
    CFG['dir_now'] = dir_now                    # 文件保存地址，放进CFG中便于
    isExists = os.path.exists(dir_now)          # 判断地址是否存在，该函数返回bool变量
    if not isExists:  os.mkdir(dir_now)         # 如果地址不存在，则创建

    # 判断存储正确率的ecel是否存在 ####################
    path = dir_now + '/测试结果.xls'
    excel = xlwt.Workbook() # 创建工作簿，用于保存实验结果
    sheet_tar = excel.add_sheet(u'acc_test_tar', cell_overwrite_ok=True) # 创建sheet，保存目标域实验结果    
    sheet_src = excel.add_sheet(u'acc_test_src', cell_overwrite_ok=True) # 创建sheet，保存源域实验结果 
    excel_files = [excel,sheet_src,sheet_tar]    

    # 判断画图保存路径是否存在
    path_name_curve = dir_now + '/' + '曲线图'     # 创建曲线图的保存位置
    isExists=os.path.exists(path_name_curve)      # 判断地址是否存在，
    if not isExists:  os.mkdir(path_name_curve)   # 如果地址不存在，则创建       

    # 选择迁移模型 ####################

    tranmodels = [
   'DDC_MSTVM',
   'RevGrad_MSTVM',
    ]
       
    common_name = '_' + srcdata_pool[ie] + '_epoch0to19_sample0to2000_cuda_rand' +str(rand_seed1) #预训练模型,文件名对应修改
    model_choosed =  model_index       # 选择用于遍历的模型index
    for _,im in enumerate(model_choosed):         # 遍历所有模型
        
        filename = 'pretrained_models' + '_rand' +str(rand_seed1) + '/'      
        CFG['tranmodel'] = tranmodels[im]         # 经典的DDA模型
        CFG['pt_name']  = filename + '[STI]_NetConv1d_MSTVM' + common_name

    
      
        main(loaders, final_test_loaders, excel_files, im)     # 调用主函数， # 参数依次是训练与测试数据集加载器，最终测试集（20个测试集，每个测试集300个样本）加载器，excel文件，模型index









