import mmd
import torch
from collections import Counter
from mmd import *


# 此函数在新版本程序中不再使用
def feamap_segmenter(feamap_src,feamap_tar,sppackt):
    feamap_seg_src, feamap_seg_tar = [],[]          # 分段后的feamap的容器
    n_in = sppackt['mulscaclf_struct'][0][0]              # 输入的样本维数，相当于每个片段的长度
    if sppackt['seg_mode'] == 'parallel':
        for ic in range(sppackt['seg_num']):      # 遍历每个分类器
            feamap_seg_src.append(feamap_src[:, 0+ic*n_in:(ic+1)*n_in])      # 分段后的feamap
            feamap_seg_tar.append(feamap_tar[:, 0+ic*n_in:(ic+1)*n_in])
    if sppackt['seg_mode'] == 'central':
        center = int(len(feamap_src[0,:]) / 2 )   # 中心点
        for ic in range(sppackt['seg_num']):              # 遍历每个分类器
            n_in_half = int( sppackt['mulscaclf_struct'][ic][0] / 2)
            feamap_seg_src.append(feamap_src[:, center-n_in_half:center+n_in_half])      # 分段后的feamap
            feamap_seg_tar.append(feamap_tar[:, center-n_in_half:center+n_in_half])
    if sppackt['seg_mode'] == 'sliding_central':
        step = int((len(feamap_src[0,:]) - n_in) / sppackt['seg_num'])   # 步长
        for ic in range(sppackt['seg_num']):              # 遍历每个分类器
            feamap_seg_src.append(feamap_src[:, 0+ic*step:n_in+ic*step])      # 分段后的feamap
            feamap_seg_tar.append(feamap_tar[:, 0+ic*step:n_in+ic*step])
    if sppackt['seg_mode'] == 'padding_central':
        len_copy = feamap_src.shape[1]
        feamap_src_copyed = torch.cat((feamap_src[:,0:len_copy],feamap_src[:,:],feamap_src[:,len_copy:]), 1)
        feamap_tar_copyed = torch.cat((feamap_tar[:,0:len_copy],feamap_tar[:,:],feamap_tar[:,len_copy:]), 1)
        center = int(len(feamap_src_copyed[0,:]) / 2 )   # 中心点
        for ic in range(sppackt['seg_num']):              # 遍历每个分类器
            n_in_half = int( sppackt['mulscaclf_struct'][ic][0] / 2)
            feamap_seg_src.append(feamap_src_copyed[:, center-n_in_half:center+n_in_half])      # 分段后的feamap
            feamap_seg_tar.append(feamap_tar_copyed[:, center-n_in_half:center+n_in_half])
    return feamap_seg_src, feamap_seg_tar





# 投票网络的函数    
def voter_multi_classifier(clfout_multi_src,clfout_multi_tar,   label_src,label_tar,   corr_multi_src,corr_multi_tar,  corr_voted_src, corr_voted_tar):
    loss_clf_fun = torch.nn.CrossEntropyLoss()
    loss_multi_clsclf_src, loss_multi_clsclf_tar = [], []
    loss_clsclf_src, loss_clsclf_tar = 0, 0
    pred_4vote_src = label_src.reshape(len(label_src),1) * 0     # 构建用于投票的pred容器
    pred_4vote_tar = label_tar.reshape(len(label_tar),1) * 0
    pred_voted_src = label_src * 0     # 构建用于投票后的pred容器
    pred_voted_tar = label_tar * 0                
    # 遍历每个用于投票的顶端分类器
    for ic in range(len(clfout_multi_src)):
        pred_src = torch.max(clfout_multi_src[ic], 1)[1]        # 计算当前投票网络的输出标签  torch.max返回的第一个是最大值，第二个是最大值的索引
        pred_tar = torch.max(clfout_multi_tar[ic], 1)[1]   
        pred_4vote_src = torch.cat([pred_4vote_src, pred_src.reshape(len(pred_src),1)],1)   # 将每个投票网络的投票结构组合进pred容器中
        pred_4vote_tar = torch.cat([pred_4vote_tar, pred_tar.reshape(len(pred_tar),1)],1)
               
        corr_multi_src[ic] = corr_multi_src[ic] + torch.sum(pred_src == label_src)    
        corr_multi_tar[ic] = corr_multi_tar[ic] + torch.sum(pred_tar == label_tar) 
        
        loss_clsclf_src_temp = loss_clf_fun(clfout_multi_src[ic], label_src)      
        loss_clsclf_tar_temp = loss_clf_fun(clfout_multi_tar[ic], label_tar)  
        loss_clsclf_src += loss_clsclf_src_temp
        loss_clsclf_tar += loss_clsclf_tar_temp
        loss_multi_clsclf_src.append(loss_clsclf_src)
        loss_multi_clsclf_tar.append(loss_clsclf_tar)
        
    # 遍历每个用于投票的pred
    pred_4vote_src = pred_4vote_src[:,1:].cpu().numpy() # 去掉第一列，第一列是全0
    pred_4vote_tar = pred_4vote_tar[:,1:].cpu().numpy()
    for ip in range(len(pred_4vote_src)):   # 遍历所有样本
        pred_voted_src[ip] = torch.tensor(Counter(pred_4vote_src[ip,:]).most_common(1)[0][0])
        pred_voted_tar[ip] = torch.tensor(Counter(pred_4vote_tar[ip,:]).most_common(1)[0][0])
    corr_voted_src += torch.sum(pred_voted_src == label_src)  
    corr_voted_tar += torch.sum(pred_voted_tar == label_tar)     
    return  corr_multi_src,corr_multi_tar,    corr_voted_src,corr_voted_tar,   loss_multi_clsclf_src,loss_multi_clsclf_tar,  loss_clsclf_src,loss_clsclf_tar

    

    
# RevGrad的梯度翻转函数    
from torch.autograd import Function
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None    
    
    
# RevGrad辅助函数    
def RevGrad_assistant(domain_classifier,feamap,label,flag):
    loss_clf_fun = torch.nn.CrossEntropyLoss()
    #loss_clf_fun = torch.nn.BCELoss()
    feamap_rev = feamap           # 容器，用于盛装梯度颠倒后的feamap
    
    for i in range(len(feamap)): 
        feamap_rev[i] = ReverseLayerF.apply(feamap[i], 1)      # 梯度反转
    domclfout = domain_classifier(feamap_rev)  # 域分类器的输出
    
    domlabel = label * 0  + flag                        # 源域的域标签
    #domlabel.float()
    loss_domclf = 0
    loss_multi_domclf = []
    for i in range(len(domclfout)):    # 遍历域判别器所有的输出，有的是1个，有的是5个
        loss_domclf_temp = loss_clf_fun(domclfout[i],domlabel)
        loss_domclf += loss_domclf_temp
        loss_multi_domclf.append(loss_domclf_temp)
        
    return loss_domclf, loss_multi_domclf



    


# 计算mmd与coral的函数
def loss_adapt_fun(X, Y, loss_tran_type):
    if loss_tran_type == 'mk_mmd':
        #mmd_loss = mmd.MMD_loss(kernel_type='rbf', kernel_mul=1, kernel_num=10)
        #loss = mmd_loss(X, Y)
        loss = mmd.mmd_rbf_noaccelerate(X, Y, kernel_mul=2.0, kernel_num=5, fix_sigma=None)
    elif loss_tran_type == 'mmd':
        #mmd_loss = mmd.MMD_loss(kernel_type='rbf', kernel_mul=1, kernel_num=1)
        #loss = mmd_loss(X, Y)  
        loss = mmd.mmd_rbf_noaccelerate(X, Y, kernel_mul=2.0, kernel_num=1, fix_sigma=None)
        #loss = mmd.mmd_rbf_loss(X, Y)

    else:
        loss = 0
    return loss
    
    
    
    