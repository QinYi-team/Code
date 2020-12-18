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
import datetime
import time
from assistant_utils import feamap_segmenter,voter_multi_classifier,ReverseLayerF,RevGrad_assistant,loss_adapt_fun
from test_utils import test
import torch.nn.functional as F   
from config import CFG
DEVICE = CFG['DEVICE']
#DEVICE = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')

# 训练函数
def train(loaders, model, optimizer,scheduler, CFG):     
    acc_test_tar_max = 0   # 最大的目标域测试正确率，用于在最大测试正确率的时候保存pt文件
    train_loader_src,  test_loader_src,   train_loader_tar,   test_loader_tar = loaders     # 样本加载器
    len_src_loader,    len_tar_loader  = len(train_loader_src), len(train_loader_tar)       # 样本中的minibatch的数目
    train_loss_clf_src,    train_loss_clf_tar,    train_loss_tran,  train_loss_total = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()    # 用于存储误差
    acc_collection, loss_collection,acc_collection_test, loss_collection_test, assistant_collection = [],[],[],[],[]       # 用于收集整个过程中的正确率与误差
    loss_24, loss_24_epoch, loss_24_collection,loss_24_collection_test = [],[],[],[]   # 24个loss保存容器

    
    #for ep in range(CFG['epoch']):        # 遍历每一个epoch
    for ep in range(50):  # 遍历每一个epoch
        # labda随着epoch调整
        labda1 = 2 / (1 + math.exp(-10 * (ep) / CFG['epoch'])) - 1    # 随着ep的变化，labda会变化
        CFG['lambda'] = 1*labda1          # 调整分类损失和迁移损失之间的动态平衡  
        # 学习速率衰减
        if ep>50:             # 设置学习速率从epoch=0开始衰减
            if ('RevGrad' in CFG['tranmodel']) :   # RevGrad模型单独考虑
                scheduler[0].step()    # FE+TC的学习速率
                scheduler[1].step()    # DomClf的学习速率
            else:
                scheduler.step()       # FE+TC的学习速率

        model.train()                     # 开始训练，向模型申明目前是训练阶段，
        # 训练监测指标的存储器
        corr_src,corr_tar = 0,0           # 正确样本数统计，对于没有投票功能的网络模型
        corr_multi_src, corr_multi_tar = [0,0,0,0,0], [0,0,0,0,0]               # 正确样本数统计，对于有投票多分类器的网络        
        corr_voted_src, corr_voted_tar = 0, 0                                   # 投票结果的正确数统计
        loss_tran_multi, loss_clf_multi_src,loss_sum = [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]        # loss分支记录
        # 样本生成器
        iter_src, iter_tar = iter(train_loader_src), iter(train_loader_tar)     # 样本batch的迭代器
        n_batch = min(len_src_loader, len_tar_loader)       # batch数，当源域与目标域的batch数目不一样的时候使用
        # 各类损失函数
        loss_clf_fun = torch.nn.CrossEntropyLoss()          # 交叉熵损失函数
        #loss_clf_fun = F.nll_loss                          # CrossEntropyLoss()=log_softmax() + NLLLoss() 
        loss_mse_fun = torch.nn.MSELoss()                   # MSE损失函数
        time_cost = 0                                       # 训练耗时
        
        for ib in range(n_batch):                      # 遍历所有n_batch
            data_src, label_src = iter_src.next()      # 训练与测试样本的迭代器
            data_tar, label_tar = iter_tar.next()
            data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)   # 将当前batch拷贝到指定的运算设备上
            data_tar, label_tar = data_tar.to(DEVICE), label_tar.to(DEVICE)   # 每次只拷贝1个batch的样本进GPU，避免占用GPU内存过大

            if ('RevGrad' in CFG['tranmodel']) :  #
                optimizer[0].zero_grad()        # 每个batch结束后都对梯度进行一次清理
                optimizer[1].zero_grad()        # 每个batch结束后都对梯度进行一次清理
            else:
                optimizer.zero_grad()        # 每个batch结束后都对梯度进行一次清理

            clfout_src, feamap_src = model(data_src)  # 源域与目标域样本模型前馈
            clfout_tar, feamap_tar = model(data_tar)
            model.train()
            ###############################################
            start_time = time.clock()             # 训练开始时间

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
                loss =loss_clsclf_src + CFG['lambda'] * loss_tran

                loss_multi_tran = []
                for il in range(len(loss_multi_domclf_src)):
                    loss_multi_tran_temp = loss_multi_domclf_src[il] + loss_multi_domclf_tar[il]
                    loss_multi_tran.append(loss_multi_tran_temp)                
                loss_24 = [loss,0,0,0,0,0,   loss_clsclf_src] + loss_multi_clsclf_src +  [loss_clsclf_tar] + loss_multi_clsclf_tar  +  [loss_tran] + loss_multi_tran                

            ###############################################    
            ''' 汇总 '''
            ###############################################            
            # 损失函数反向传播
            # 损失函数反向传播
            if ('RevGrad' in CFG['tranmodel']):
                loss_domclf.backward(retain_graph=True)       # 域判别器  误差反向传播，求梯度
                optimizer[1].step()      # 域判别器  误将delta更新在权值与偏置值上
                loss.backward()       # 主干部分 误差反向传播，求梯度
                optimizer[0].step()      # 主干部分 将delta更新在权值与偏置值上
            else:
                loss.backward()       # 误差反向传播，求梯度
                optimizer.step()      # 将delta更新在权值与偏置值上
            
            # 时间结算
            end_time = time.clock()                # 训练结束时间
            time_cost += end_time - start_time          # 累计每一个batch的训练耗时
                            
            train_loss_clf_src.update(loss_clsclf_src.item())     # 在loss的和上，加一个值
            train_loss_clf_tar.update(loss_clsclf_tar.item())
            train_loss_tran.update(loss_tran.item())
            train_loss_total.update(loss.item())
            if ib % CFG['log_interval'] == 0:     # 每隔多少个batch停顿一下           
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.4f}, loss_tran: {:.4f}, total_Loss: {:.4f}'
                      .format(ep + 1,CFG['epoch'],int(100.*ib/n_batch), train_loss_clf_src.avg, train_loss_tran.avg, train_loss_total.avg))

            # loss统计    
            for i24 in range(24):
                loss_24[i24] = loss_24[i24].item()  if loss_24[i24]!=0  else  loss_24[i24]    # 三目运算符
                            ## 如果loss_24[i24]!=0，则表示loss_24[i24]是touch.tensor，因此使用.item()来将其转换为普通float； 如果loss_24[i24]==0，则表明其只是一个填充占位符，因此直接用
            loss_24_epoch.append(loss_24)   # # loss_24_epoch中记录着本格epoch中所有batch计算所得的24个loss，每一行就是一个batch所计算出的loss
        loss_24_epoch_avg = []          # 求取平均损失的容器
        num_batch = len(loss_24_epoch)  # batch数
        for i1 in range(24):            # 遍历24个loss值，    
            loss_temp = 0
            for i2 in range(num_batch):    # 遍历每个batch
                loss_temp += loss_24_epoch[i2][i1]   # 把每个loss值沿着batch方向求和
            loss_temp_avg = loss_temp / num_batch    # 求每个batch的平均值   
            loss_24_epoch_avg.append(loss_temp_avg)  # 本次epoch下的平均损失      
        loss_24_collection.append(loss_24_epoch_avg) # 将每个epoch下的24个平均损失收集起来   
        


        multi_acc_src, multi_acc_tar = [],[]         # 多分类器的正确率容器
        for ic in range(5):       # 遍历5个分支分类器
            acc_src_temp = 100. * corr_multi_src[ic].cpu().numpy() / len(train_loader_src.dataset)   # 利用分类正确的样本数计算正确率
            acc_tar_temp = 100. * corr_multi_tar[ic].cpu().numpy() / len(train_loader_tar.dataset)
            multi_acc_src.append(acc_src_temp)   # multi_acc_src是个list，里面存放五个分支的正确率
            multi_acc_tar.append(acc_tar_temp)
        acc_voted_src = 100. * corr_voted_src.cpu().numpy() / len(train_loader_src.dataset)    # 投票后的正确率
        acc_voted_tar = 100. * corr_voted_tar.cpu().numpy() / len(train_loader_tar.dataset)
        # 要return的结果
        acc_all = [acc_voted_src] + multi_acc_src + [acc_voted_tar] + multi_acc_tar       # 将训练所得的所有正确率汇集到一起
        acc_collection.append(acc_all)      # # 从左到右依次是：1个源域投票正确率、5个源域分支正确率、1个目标域投票正确率、5个目标域分支正确率
        loss_all = [train_loss_clf_src.avg, train_loss_clf_tar.avg, train_loss_tran.avg, train_loss_total.avg]
        loss_collection.append(loss_all)    # 从左到右依次是：源域测试集分类损失，目标域测试集分类损失，测试集迁移损失，测试集总损失
        # 屏幕打印
        print(multi_acc_src)   # 打印当前epoch下的5个分支源域正确率
        print(multi_acc_tar)   # 打印当前epoch下的5个分支目标域正确率
        print([acc_voted_src,acc_voted_tar])   # 打印源域和目标域投票后的正确率
        print('Train: source_acc:{:.2f}%({}/{}),  target_acc:{:.2f}%({}/{})\n'.format(acc_voted_src,corr_voted_src,len(train_loader_src.dataset),   acc_voted_tar,corr_voted_tar,len(train_loader_tar.dataset)))
        print('########\n')    
        
              
              
        # 测试集返回
        acc_collection_test_temp, loss_collection_test_temp ,loss_24_epoch_avg_temp = test(model, test_loader_src, test_loader_tar, ep, CFG)
        acc_collection_test.append(acc_collection_test_temp)     # 将测试正确率qppend进去
        loss_collection_test.append(loss_collection_test_temp)   # 将测试损失append进去
        loss_24_collection_test.append(loss_24_epoch_avg_temp)
        if ('RevGrad' in CFG['tranmodel']) or ('DCTLN' in CFG['tranmodel']):
            assistant_collection.append([scheduler[0].get_lr()[0],CFG['lambda'],time_cost])  # scheduler[0]yu scheduler[1]的lr衰减速率相同
        else:
            assistant_collection.append([scheduler.get_lr()[0],CFG['lambda'],time_cost])  # 将辅助记录append进去
        
        # 判断当前是否保存pt文件
        result_trace = [acc_collection,loss_collection,  acc_collection_test,loss_collection_test,  assistant_collection]   # 所有的记录合并在一起    
        acc_test_tar_current = acc_collection_test_temp[6]   # 当前的测试正确率是第7个，也就是前面6个是训练正确率
        if acc_test_tar_current > acc_test_tar_max:     # 如果当前正确率大于最大正确率
            acc_test_tar_max = acc_test_tar_current    # 将最大正确率交换成当前正确率
            if ('RevGrad' in CFG['tranmodel']):
                pt = {'model':model.state_dict(), 'optimizer0':optimizer[0].state_dict(), 'optimizer1':optimizer[1].state_dict(),'scheduler0':scheduler[0].state_dict(),'scheduler1':scheduler[1].state_dict(), 'CFG':CFG,  'result_trace':result_trace}
            else:
                pt = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(), 'CFG':CFG,  'result_trace':result_trace}
            torch.save(pt,  CFG['dir_now'] + '/' + CFG['tranmodel']  +'_maxacc.pt')         # 并且，保存当前正确率下的模型    
            
    trace = [acc_collection,loss_collection,loss_24_collection,   acc_collection_test,loss_collection_test,loss_24_collection_test,  assistant_collection]    # 记录器
    return trace




