import numpy as np
import torch
import torch.nn as nn
#import torchvision
#from torchvision import models
from torch.autograd import Variable
import copy


# ===================================================================
# 一维正向卷积模块，用于构建FE的各层
def BNConv1dReLU(in_channels,out_channels,kernel_size, stride=1,  padding=0):  # 
    return nn.Sequential(
        nn.BatchNorm1d(in_channels),     # 批规范化
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),  # 一维卷积
        nn.ReLU(inplace=True),)     # 激活函数

# ===================================================================
# 全连接模块，用于构建顶部特征提取器    
def BNFCLReLU(in_size,out_size):
    return nn.Sequential(
        nn.BatchNorm1d(in_size), 
        nn.Linear(in_size,out_size),  # 全连接层 fullconnected layers
        nn.ReLU6(inplace=True),)    

# ===================================================================
# Flatten继承Module
class Flatten(nn.Module):
    def __init__(self):                             # 构造函数，没有什么要做的
        super(Flatten, self).__init__()             # 调用父类构造函数
    def forward(self, input):                       # 实现forward函数    
        return input.view(input.size(0), -1)        # 保存batch维度，后面的维度全部压平，例如输入是28*28的特征图，压平后为784的向量



# ===================================================================
# ===================================================================
# ===================================================================

class Mid_MSTM_Icp(nn.Module):    # 针对NetConv1d_3Icp_Vote网络，
    def __init__(self):
        super(Mid_MSTM_Icp, self).__init__()
        self.Mid_name = 'Mid_MSTM_Icp'
        N_inCh = 128    # 所有分支输入的channel是相同
        N_hidCh = [8,8,16,32,64]
        N_outCh = [8,8,16,32,64]
        self.branch1 = nn.Sequential(BNConv1dReLU(in_channels=N_inCh,  out_channels=N_hidCh[0],  kernel_size=1,  stride=1,  padding=0),  # 一维卷积压缩通道
                                     nn.MaxPool1d(kernel_size=3, stride=3, padding=0),
                                     Flatten() )
        self.branch2 = nn.Sequential(BNConv1dReLU(in_channels=N_inCh,  out_channels=N_hidCh[1],  kernel_size=1,  stride=1,  padding=0),
                                     nn.MaxPool1d(kernel_size=4, stride=3, padding=0),
                                     Flatten() )
        self.branch3 = nn.Sequential(BNConv1dReLU(in_channels=N_inCh,  out_channels=N_hidCh[2],  kernel_size=1,  stride=1,  padding=0),
                                     BNConv1dReLU(in_channels=N_hidCh[2],  out_channels=N_outCh[2],  kernel_size=4,  stride=3,  padding=0),
                                     Flatten() )
        self.branch4 = nn.Sequential(BNConv1dReLU(in_channels=N_inCh,  out_channels=N_hidCh[3],  kernel_size=1,  stride=1,  padding=0),
                                     BNConv1dReLU(in_channels=N_hidCh[3],  out_channels=N_outCh[3],  kernel_size=3,  stride=3,  padding=0),
                                     Flatten() )
        self.branch5 = nn.Sequential(BNConv1dReLU(in_channels=N_inCh,  out_channels=N_hidCh[4],  kernel_size=1,  stride=1,  padding=0),
                                     BNConv1dReLU(in_channels=N_hidCh[4],  out_channels=N_outCh[4],  kernel_size=2,  stride=4,  padding=0),
                                     Flatten() )
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out5 = self.branch5(x)
        feamap = [out1, out2, out3, out4, out5]    # 用作迁移特征
        midout = feamap    # 分类器为全连接
        midout_concat = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return feamap, midout, midout_concat       #


#  基本思想，基于Inception构建，在FE-4输出Feature后，利用多尺度的Mid模块将其分解为5个多尺度的TF，然后分别输入到5个不同的TC中
class Clf_MSTVM(nn.Module):
    def __init__(self):
        super(Clf_MSTVM, self).__init__()
        self.Clf_name = 'Clf_MSTVM'
        N_size = [i*4 for i in [8,8,16,32,64]]

        self.branch1 = nn.Sequential(BNFCLReLU(N_size[0], 5),)
                                     #nn.Softmax(dim=1),)     
        self.branch2 = nn.Sequential(BNFCLReLU(N_size[1], 5),)
                                     #nn.Softmax(dim=1), )                                 
        self.branch3 = nn.Sequential(BNFCLReLU(N_size[2], 5),)
                                     #nn.Softmax(dim=1),)          
        self.branch4 = nn.Sequential(BNFCLReLU(N_size[3], 64),
                                     nn.BatchNorm1d(64),
                                     nn.Linear(64,5), )
                                     #nn.Softmax(dim=1),)      
        self.branch5 = nn.Sequential(BNFCLReLU(N_size[4], 64),
                                     nn.BatchNorm1d(64),
                                     nn.Linear(64,5), )
                                     #nn.Softmax(dim=1),)          
    def forward(self, x):
        out1 = self.branch1(x[0])
        out2 = self.branch2(x[1])
        out3 = self.branch3(x[2])
        out4 = self.branch4(x[3])
        out5 = self.branch5(x[4])        
        Clf_softmax = nn.Softmax(dim=1)    # 借用softmax将进行投票统计
        out_sum =  out1 + out2 + out3 + out4 + out5
        out_voted = Clf_softmax(out_sum)
        out = [ out_voted, out1, out2, out3, out4, out5]
        return out

  
class NetConv1d_MSTVM(nn.Module):   
    def __init__(self,CFG):
        super(NetConv1d_MSTVM, self).__init__()
        self.name = 'NetConv1d_MSTVM'   
        self.num_class = CFG['num_class']
        N_FECh = [1,32,64,128,128,128,128]
        self.N_FECh = N_FECh
        # 卷积与池化的计算公式  Hout = （Hin - Kernel + 2*padding）/stride + 1

        self.FE_block1 = nn.Sequential(BNConv1dReLU(in_channels=N_FECh[0],  out_channels=N_FECh[1],  kernel_size=128,  stride=1,  padding=0),   # 2945
                                       nn.MaxPool1d(kernel_size=4, stride=4, padding=0) )
        self.FE_block2 = nn.Sequential(BNConv1dReLU(in_channels=N_FECh[1],  out_channels=N_FECh[2],  kernel_size=32,  stride=1,  padding=0),    # 950
                                       nn.MaxPool1d(kernel_size=4, stride=4, padding=0) )
        self.FE_block3 = nn.Sequential(BNConv1dReLU(in_channels=N_FECh[2],  out_channels=N_FECh[3],  kernel_size=8,  stride=1,  padding=0),     # 309
                                       nn.MaxPool1d(kernel_size=4, stride=4, padding=0) )
        self.FE_block4 = nn.Sequential(BNConv1dReLU(in_channels=N_FECh[3],  out_channels=N_FECh[4],  kernel_size=3,  stride=1,  padding=1),     # 101
                                     nn.MaxPool1d(kernel_size=3, stride=3, padding=0) )

        self.Mid_block1 = nn.Sequential(Mid_MSTM_Icp() )

        self.Clf_block1 = Clf_MSTVM()
        
    def forward(self, x):
        x = torch.reshape(x, (-1,1,3072))
        x = self.FE_block1(x)
        x = self.FE_block2(x) 
        x = self.FE_block3(x)      
        x = self.FE_block4(x)    

        feamap,x,_ = self.Mid_block1(x) 
        output = self.Clf_block1(x)
        return output, feamap 


# ===================================================================
'''五个TF的域判别器'''      #
class DomClf_FCL_5(nn.Module):
    def __init__(self):
        super(DomClf_FCL_5, self).__init__()
        self.DomClf_name = 'DomClf_FCL_5'
        N_size = [i*4 for i in [8,8,16,32,64]]   # 每个channel有16个参数，N_size表格各层全连接的size
        self.branch1 = nn.Sequential(nn.BatchNorm1d(N_size[0]),nn.Linear(N_size[0],2), nn.Softmax(dim=1))         
        self.branch2 = nn.Sequential(nn.BatchNorm1d(N_size[1]),nn.Linear(N_size[1],2), nn.Softmax(dim=1))         
        self.branch3 = nn.Sequential(nn.BatchNorm1d(N_size[2]),nn.Linear(N_size[2],2), nn.Softmax(dim=1))         
        self.branch4 = nn.Sequential(BNFCLReLU(N_size[3], 128), nn.BatchNorm1d(128),nn.Linear(128,2), nn.Softmax(dim=1)) 
        self.branch5 = nn.Sequential(BNFCLReLU(N_size[4], 128), nn.BatchNorm1d(128),nn.Linear(128,2), nn.Softmax(dim=1)) 
        self.branch1 = nn.Sequential(nn.BatchNorm1d(N_size[0]),nn.Linear(N_size[0],2))         
        self.branch2 = nn.Sequential(nn.BatchNorm1d(N_size[1]),nn.Linear(N_size[1],2))       
        self.branch3 = nn.Sequential(nn.BatchNorm1d(N_size[2]),nn.Linear(N_size[2],2))         
        self.branch4 = nn.Sequential(BNFCLReLU(N_size[3], 128), nn.BatchNorm1d(128),nn.Linear(128,2)) 
        self.branch5 = nn.Sequential(BNFCLReLU(N_size[4], 128), nn.BatchNorm1d(128),nn.Linear(128,2))        
    def forward(self, x):
        out1 = self.branch1(x[0])
        out2 = self.branch2(x[1])
        out3 = self.branch3(x[2])
        out4 = self.branch4(x[3])
        out5 = self.branch5(x[4])     
        return [out1, out2, out3, out4, out5]


