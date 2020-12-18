# 测试函数
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

from config import CFG
DEVICE = CFG['DEVICE']
#DEVICE = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')



def test(model, test_loader_src, test_loader_tar, ep, CFG):
    model.eval()        # 开启测试
    
    len_src_loader,    len_tar_loader  = len(test_loader_src), len(test_loader_tar)      # 测试集中的batch数
    test_loss_clf_src,    test_loss_clf_tar,    test_loss_tran,  test_loss_total = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
    acc_collection = []       # 用于收集整个过程中的正确率
    loss_collection = []    
    loss_24, loss_24_epoch, loss_24_collection,loss_24_collection_test = [],[],[],[]    

    corr_src,corr_tar = 0,0     # 正确数
    corr_multi_src, corr_multi_tar = [0,0, 0,0,0], [0,0, 0,0,0]         # 分类器的正确数
    corr_voted_src, corr_voted_tar = 0, 0                               # 投票结果的正确数
    iter_src, iter_tar = iter(test_loader_src), iter(test_loader_tar)   # 生成器
    n_batch = min(len_src_loader, len_tar_loader)                       # 最小batch数
    loss_clf_fun = torch.nn.CrossEntropyLoss()
    loss_mse_fun = torch.nn.MSELoss()    
    
    with torch.no_grad():              # 每隔batch清除梯度           
        for ib in range(n_batch):      # 遍历所有的batch
            data_src, label_src = iter_src.next()      # 迭代器
            data_tar, label_tar = iter_tar.next()    
            data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)    # 拷贝到指定的设备
            data_tar, label_tar = data_tar.to(DEVICE), label_tar.to(DEVICE)
            
            clfout_src,feamap_src = model(data_src)            # 网络前馈
            clfout_tar,feamap_tar = model(data_tar)
            
            ###############################################       

            




            ###############################################       
            ''' MSTVM 投票网络'''
            ###############################################

            if CFG['tranmodel']=='DDC_MSTVM':                
                loss_tran = 0
                loss_multi_tran = []
                for ic in range(len(feamap_src)):      # 遍历每个灯笼分支
                    loss_tran_temp = loss_adapt_fun(feamap_src[ic],feamap_tar[ic],'mmd') 
                    loss_tran += loss_tran_temp 
                    loss_multi_tran.append(loss_tran_temp)
                    
                return_packt = voter_multi_classifier(clfout_src[1:], clfout_tar[1:], label_src,label_tar, corr_multi_src,corr_multi_tar, corr_voted_src,corr_voted_tar)
                corr_multi_src,corr_multi_tar,  corr_voted_src,corr_voted_tar,   loss_multi_clsclf_src,loss_multi_clsclf_tar,   loss_clsclf_src,loss_clsclf_tar = return_packt   
                    
                loss = loss_clsclf_src + CFG['lambda'] * loss_tran   

                loss_24 = [loss,0,0,0,0,0,   loss_clsclf_src] + loss_multi_clsclf_src +  [loss_clsclf_tar] + loss_multi_clsclf_tar  +  [loss_tran] + loss_multi_tran

            ###############################################                 
            if CFG['tranmodel']=='RevGrad_MSTVM': 
                loss_domclf_src, loss_multi_domclf_src = RevGrad_assistant(model.DomClf_net,feamap_src,label_src,0)    # 梯度翻转的辅助函数
                loss_domclf_tar, loss_multi_domclf_tar = RevGrad_assistant(model.DomClf_net,feamap_tar,label_tar,1)
                loss_tran = loss_domclf_src + loss_domclf_tar
                
                return_packt = voter_multi_classifier(clfout_src[1:], clfout_tar[1:], label_src,label_tar, corr_multi_src,corr_multi_tar, corr_voted_src,corr_voted_tar)
                corr_multi_src,corr_multi_tar,  corr_voted_src,corr_voted_tar,   loss_multi_clsclf_src,loss_multi_clsclf_tar,   loss_clsclf_src,loss_clsclf_tar = return_packt
                loss_domclf = loss_tran     # 用于domclf的单独参数更新
                loss = loss_clsclf_src + CFG['lambda'] * loss_tran

                loss_multi_tran = []
                for il in range(len(loss_multi_domclf_src)):
                    loss_multi_tran_temp = loss_multi_domclf_src[il] + loss_multi_domclf_tar[il]
                    loss_multi_tran.append(loss_multi_tran_temp)                
                loss_24 = [loss,0,0,0,0,0,   loss_clsclf_src] + loss_multi_clsclf_src +  [loss_clsclf_tar] + loss_multi_clsclf_tar  +  [loss_tran] + loss_multi_tran                

            ###############################################    
            ''' 汇总 '''
            ###############################################           
            test_loss_clf_src.update(loss_clsclf_src.item())     # 将最新的loss值加进去
            test_loss_clf_tar.update(loss_clsclf_tar.item())
            test_loss_tran.update(loss_tran.item())
            test_loss_total.update(loss.item())
            if ib % CFG['log_interval'] == 0:           # batch间隔打印     
                print('Test Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.4f}, loss_tran: {:.4f}, total_Loss: {:.4f}'
                      .format(ep + 1,CFG['epoch'],int(100.*ib/n_batch), test_loss_clf_src.avg, test_loss_tran.avg, test_loss_total.avg))                    
            
            # loss全方位追踪   
            for i24 in range(24):    # 一共追踪24个loss
                loss_24[i24] = loss_24[i24].item()  if loss_24[i24]!=0  else  loss_24[i24]    # 三目运算符，
                       # 如果loss_24[i24]!=0，则表示loss_24[i24]是touch.tensor，因此使用.item()来将其转换为普通float； 如果loss_24[i24]==0，则表明其只是一个填充占位符，因此直接用
            loss_24_epoch.append(loss_24)   # loss_24_epoch中记录着本格epoch中所有batch计算所得的24个loss，每一行就是一个batch所计算出的loss
        loss_24_epoch_avg = []          # 求取平均损失的容器
        num_batch = len(loss_24_epoch)  # batch数
        for i1 in range(24):            # 遍历24个loss值，
            loss_temp = 0           
            for i2 in range(num_batch):     # 遍历每个batch
                loss_temp += loss_24_epoch[i2][i1]   # 把每个loss值沿着batch方向求和
            loss_temp_avg = loss_temp / num_batch    # 求每个batch的平均值   
            loss_24_epoch_avg.append(loss_temp_avg)  # 本次epoch下的平均损失  


        if 'TVM' in CFG['tranmodel']:    # 如果模型中涉及到投票机制，那就需要引入下面的投票器
            multi_acc_src, multi_acc_tar = [],[]             # 多分类器的正确率容器
            for ic in range(5):             # 遍历5个分支分类器
                acc_src_temp = 100. * corr_multi_src[ic].cpu().numpy() / len(test_loader_src.dataset)  # 利用分类正确的样本数计算正确率
                acc_tar_temp = 100. * corr_multi_tar[ic].cpu().numpy() / len(test_loader_tar.dataset)   
                multi_acc_src.append(acc_src_temp)         # multi_acc_src是个list，里面存放五个分支的正确率
                multi_acc_tar.append(acc_tar_temp)
            acc_voted_src = 100. * corr_voted_src.cpu().numpy() / len(test_loader_src.dataset)   # 投票后的正确率
            acc_voted_tar = 100. * corr_voted_tar.cpu().numpy() / len(test_loader_tar.dataset)  
            # 要return的结果
            acc_all = [acc_voted_src] + multi_acc_src + [acc_voted_tar] + multi_acc_tar       # 将训练所得的所有正确率汇集到一起，
            acc_collection = acc_all     # 从左到右依次是：1个源域投票正确率、5个源域分支正确率、1个目标域投票正确率、5个目标域分支正确率
            loss_all = [test_loss_clf_src.avg, test_loss_clf_tar.avg, test_loss_tran.avg, test_loss_total.avg]  # 损失记录
            loss_collection = loss_all   # 从左到右依次是：源域测试集分类损失，目标域测试集分类损失，测试集迁移损失，测试集总损失
            # 屏幕打印
            print(multi_acc_src)   # 打印当前epoch下的5个分支源域正确率
            print(multi_acc_tar)   # 打印当前epoch下的5个分支目标域正确率
            print([acc_voted_src,acc_voted_tar])    # 打印源域和目标域投票后的正确率
            print('Test: source_acc:{:.2f}%({}/{}),  target_acc:{:.2f}%({}/{})\n'.format(acc_voted_src,corr_voted_src,len(test_loader_src.dataset),   acc_voted_tar,corr_voted_tar,len(test_loader_tar.dataset)))    
        else:    # 如果没有涉及到投票机制，则执行下面的条件选项
            acc_src = 100. * corr_src.cpu().numpy() / len(test_loader_src.dataset)    # 正确数除以所有的样本数，等于正确率
            acc_tar = 100. * corr_tar.cpu().numpy() / len(test_loader_tar.dataset)             
            acc_all = [acc_src] + [0,0,0,0,0] + [acc_tar] + [0,0,0,0,0]       # 将训练所得的所有正确率汇集到一起，分支正确率用0代替
            acc_collection = acc_all        # 本次测试的正确值
            loss_all = [test_loss_clf_src.avg, test_loss_clf_tar.avg, test_loss_tran.avg, test_loss_total.avg]  # 本次测试所得的损失
            loss_collection = loss_all      # 从左到右依次是：源域测试集分类损失，目标域测试集分类损失，测试集迁移损失，测试集总损失 
            print('Test: source_acc:{:.2f}%({}/{}),  target_acc:{:.2f}%({}/{})\n'.format(acc_src,corr_src,len(test_loader_src.dataset),   acc_tar,corr_tar,len(test_loader_tar.dataset))) 
        print('###############################################################\n')
    
    return acc_collection, loss_collection ,loss_24_epoch_avg         
                















    
