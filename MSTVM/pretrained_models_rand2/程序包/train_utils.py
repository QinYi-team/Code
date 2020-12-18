from __future__ import print_function          # 调用Python未来的输出函数print
import argparse                                # 参数解析模块
import torch
import torch.nn as nn
import torch.nn.functional as F   
from torch.autograd import Variable             
import math



def train(params, model, train_loader, optimizer, epoch):
    model.train()                                                  # 声明当前为训练模式
    loss_epoch_i = [0,  0,0,0,0,0]     # 当前epoch的loss和acc
    acc_epoch_i = [0,  0,0,0,0,0]     
    loss_batch = [0,  0,0,0,0,0]    # 逐个batch记录loss和corr
    corr_batch = [0,  0,0,0,0,0]
    for batch_idx, (sample, target) in enumerate(train_loader):     # 依次遍历所有的训练样本
        sample, target = sample.to(params['device']), target.to(params['device'])          # 将样本拷贝到指定设备
        optimizer.zero_grad()                                       # 训练每个batch之前都需要将loss关于weight的导数置为0，因为backward()是吧所有batch的梯度累加起来
        output,_ = model(sample)                                       # 样本前馈模型
        
        loss_batch, corr_batch, loss_4back = train_assistant(output, target, model.name,loss_batch,corr_batch) # 正确数和损失
        loss_4back.backward(retain_graph=True,create_graph=True)          # 计算每个参数的更新值delta
        optimizer.step()                                            # 将所有delta施加到参数中

    for i in range(6):     # 遍历5个分支和一个投票结果
        loss_epoch_i[i] = float(loss_batch[i]) / len(train_loader)         # 当前epoch的平均loss    len(train_loader.dataset) 是样本数，len(train_loader) 是batch数
        acc_epoch_i[i]  = 100.*corr_batch[i] / len(train_loader.dataset)    # 当前epoch下的正确率=所有batch的正确数除以训练样本数
    
    print('\nEpoch:',epoch,'======================')
    print('Train: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
          loss_epoch_i[0], corr_batch[0], len(train_loader.dataset), acc_epoch_i[0]))
    #print(loss_epoch_i)
    print(acc_epoch_i)
    return loss_epoch_i, acc_epoch_i

def test(params, model, test_loader, is_print=False):                                    # 测试函数  
    model.eval()                                                               # 测试模式 
    loss_epoch_i = [0,  0,0,0,0,0]     # 当前epoch的loss和acc
    acc_epoch_i = [0,  0,0,0,0,0]     
    loss_batch = [0,  0,0,0,0,0]    # 逐个batch记录loss和corr
    corr_batch = [0,  0,0,0,0,0]
    with torch.no_grad():                                                      # 在一下结构中，数据不计算梯度，也不进行反向传播
        for sample, target in test_loader:                                       # 遍历所有测试样本,sample与target是tensor类型，test_loader是type(test_loader)类
            sample, target = sample.to(params['device']), target.to(params['device'])                  # 样本拷贝到指定设备中 
            output,f4t = model(sample)       # output是顶部分类器的输出，f4t是Mid层输出的迁移特征       
            loss_batch, corr_batch, loss_4back = train_assistant(output, target, model.name,loss_batch,corr_batch) # 正确数和损失

        for i in range(6):     # 遍历5个分支和一个投票结果
            loss_epoch_i[i] = float(loss_batch[i]) / len(test_loader)         # 当前epoch的平均loss    len(train_loader.dataset) 是样本数，len(train_loader) 是batch数
            acc_epoch_i[i]  = 100.*corr_batch[i] / len(test_loader.dataset)    # 当前epoch下的正确率=所有batch的正确数除以训练样本数

    print('Test: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
          loss_epoch_i[0], corr_batch[0], len(test_loader.dataset), acc_epoch_i[0]))
    #print(loss_epoch_i)
    print(acc_epoch_i)
    return loss_epoch_i, acc_epoch_i,[f4t,target]

#######################################################################################################


# 用作投票辅助
def voter_assistant(output, target):
    output = output[1:]        # 投票系统的第一个是统一容器
    loss = 0                   # 用作容器
    output_sum = output[0]*0   # 用作容器
    Clf_softmax = nn.Softmax(dim=1)    # 借用softmax将输出转换成统一格式
    for i in range(5):
        loss += F.nll_loss(output[i], target)    
        output_sum += output[i]
    output = Clf_softmax(output_sum)
    return output,loss
        
        
    
# 用作投票辅助
def train_assistant(output_branch, target, model_name,loss_batch,corr_batch):
    loss_branch = [0,  0,0,0,0,0]    # 容器：损失
    pred_branch = [0,  0,0,0,0,0]    # 容器：正确结果
    corr_branch = [0,  0,0,0,0,0]    # 容器：正确数
    if 'Vote' in model_name:    # 对于投票模型
        for i in range(6):      # 遍历5个分支和一个投票结果
            loss_branch[i] = F.nll_loss(output_branch[i], target)    
            pred_branch[i] = output_branch[i].argmax(dim=1, keepdim=True)   
            corr_branch[i] = float(pred_branch[i].eq(target.view_as(pred_branch[i])).sum().item()) 
        loss_branch[0] = loss_branch[1] + loss_branch[2] + loss_branch[3] + loss_branch[4] + loss_branch[5]    # loss0是5个分支的loss相加，这样才能避免分支的学习速度跟不上
    else:            # 对于非投票模型
        loss_branch[0] = F.nll_loss(output_branch[0], target)   
        pred_branch[0] = output_branch[0].argmax(dim=1, keepdim=True)  
        corr_branch[0] = float(pred_branch[0].eq(target.view_as(pred_branch[0])).sum().item()) 
    loss_4back = loss_branch[0]     # 用于误差反向传播的loss
      
    for i in range(6):
        loss_batch[i] += loss_branch[i]
        corr_batch[i] += corr_branch[i]        
        
    return loss_batch, corr_batch, loss_4back
              
        
        
        
        
        
        
        
        
        

