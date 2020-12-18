from __future__ import print_function          # 调用Python未来的输出函数print
import argparse                                # 参数解析模块
import torch
import torch.nn as nn
import torch.nn.functional as F                
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio 
import numpy as np
import random
from sklearn import preprocessing



class MyDataset(Dataset):
    def __init__(self,dataset_choose, no_start, no_end):
        super(MyDataset,self).__init__()
        self.no_start = no_start 
        self.no_end = no_end
        self.dataset_choose = dataset_choose
        self.dataset_base_path = 'D:\PYTHON\迁移学习\王鑫-交接文件\迁移学习-交接文件\新2020-datasets'  # 样本集存放地址
        # 选择样本集  
        if self.dataset_choose[0:18] ==   'PlanetGearFaultRaw':    #  
            self.sample,self.label = self.PlanetGearFaultRaw_loader()    # 调用子函数     
        elif self.dataset_choose[0:18] == 'PlanetGearRaw4Ches':    #  
            self.sample,self.label = self.PlanetGearRaw4Ches_loader()    # 调用子函数    
        elif self.dataset_choose[0:18] == 'RealWindTurbineRaw':    #  
            self.sample,self.label = self.RealWindTurbineRaw_loader()    # 调用子函数                 
        elif self.dataset_choose[0:18] == '608BearingFaultRaw':
            self.sample,self.label = self.A608BearingFaultRaw_loader()    # 调用子函数               
        elif self.dataset_choose[0:18] == 'CasBearingFaultRaw': 
            self.sample,self.label = self.CaseBearingFaultRaw_loader()    # 调用子函数
        elif self.dataset_choose[0:18] == 'IMSBearingFaultRaw':
            self.sample,self.label = self.IMSBearingFaultRaw_loader()    # 调用子函数     
        elif self.dataset_choose[0:18] == 'ZouBearingFaultRaw':
            self.sample,self.label = self.ZouBearingFaultRaw_loader()    # 调用子函数                  

        elif self.dataset_choose[0:18] == 'PlanetGearFaultFea':
            self.sample,self.label = self.PlanetGearFaultFea_loader()    # 调用子函数

        elif self.dataset_choose[0:11] == 'SelfBearing':
            self.sample, self.label = self.SelfBearing_loader()  # 调用子函数

        self.num_sample = self.sample.shape[0]          # 样本数目
    def __len__(self):
        return self.num_sample
    def __getitem__(self,idx):
        return self.sample[idx,:],self.label[idx]


    def get_scaler(self):
        return self.scaler


##########################################################################  
    
    def RealWindTurbineRaw_loader(self):        # 专门针对行星齿轮故障信号数据集的样本加载函数
        DATA = np.load(self.dataset_base_path + '/RealWindTurbineRaw-样本拼接之后/' + self.dataset_choose  + '.npz')
        sample = DATA['sample']
        label  = DATA['label']

        sample = sample[self.no_start:self.no_end,:]  
        label  = label[self.no_start:self.no_end,:]       
        sample = np.transpose(sample)
        sample,scaler = self.sample_scaler(sample,mode=['minmax','z-score','maxabs'][1])
        sample = np.transpose(sample)
        sample = torch.tensor(sample, dtype=torch.float32)     # 将样本转换为特定格式
        label = torch.tensor(label, dtype=torch.long)
        label = label.squeeze(1)                                # 将标签转换为只有batchsize这一个维度的数据结构，不然会报错
        return sample,label      


##########################################################################      
    def PlanetGearRaw4Ches_loader(self):        # 专门针对行星齿轮故障信号数据集的样本加载函数
        DATA = np.load(self.dataset_base_path + '/PlanetGearRaw4Ches-四个通道的数据/' + self.dataset_choose  + '.npz')
        sample = DATA['sample']
        label  = DATA['label']
        load   = DATA['load']        

        sample = sample[self.no_start:self.no_end,:]  
        label  = label[self.no_start:self.no_end,:]  
        load   = load[self.no_start:self.no_end,:]       
        sample = np.transpose(sample)         # 将样本颠倒过来，对每个样本内部进行归一化
        sample,scaler = self.sample_scaler(sample,mode=['minmax','z-score','maxabs'][1])   # 归一化方式
        sample = np.transpose(sample)         # 再将样本复原
        sample = torch.tensor(sample, dtype=torch.float32)     # 将样本转换为特定格式
        label = torch.tensor(label, dtype=torch.long)
        label = label.squeeze(1)                               # 将标签转换为只有batchsize这一个维度的数据结构，不然会报错
        return sample,label      


##########################################################################      
    def PlanetGearFaultRaw_loader(self):        # 专门针对行星齿轮故障信号数据集的样本加载函数
        DATA = np.load(self.dataset_base_path + '/PlanetGearFaultRaw-原始数据与特征数据/' + self.dataset_choose  + '.npz')
        sample = DATA['sample']
        label  = DATA['label']
        load   = DATA['load']
        
        sample = sample[self.no_start:self.no_end,:]  
        label  = label[self.no_start:self.no_end,:]  
        load   = load[self.no_start:self.no_end,:]       
        sample = np.transpose(sample)
        sample,scaler = self.sample_scaler(sample,mode=['minmax','z-score','maxabs'][1])
        sample = np.transpose(sample)
        sample = torch.tensor(sample, dtype=torch.float32)     # 将样本转换为特定格式
        label = torch.tensor(label, dtype=torch.long)
        label = label.squeeze(1)                                # 将标签转换为只有batchsize这一个维度的数据结构，不然会报错
        return sample,label      




##########################################################################  
    def CaseBearingFaultRaw_loader(self):        # 专门针对行星齿轮故障信号数据集的样本加载函数
        DATA = np.load('D:\迁移学习\迁移学习-自建\故障信号\datasets/CaseBearingFaultRaw-原始数据-整理为字典/' + self.dataset_choose  + '.npz')
        sample = DATA['sample']
        label  = DATA['label']
        load   = DATA['load']
        # 数据筛选，筛选出部分特征的数据
        index_selected =  (label[:,0]<4)#&(load[:,0]==0)     # 只取前四种故障，0：normal， 1：inner rac， 2：ball， 3：outerrace-1   4：outerrace2
        sample = sample[index_selected,:]   
        label  = label[index_selected,:]
        load   = load[index_selected,:]
        # 数据量的选择，
        sample = sample[self.no_start:self.no_end,:]*1
        label  = label[self.no_start:self.no_end,:]
        load   = load[self.no_start:self.no_end,:] 
        # 数据归一化
        sample = np.transpose(sample)
        sample,scaler = self.sample_scaler(sample,mode=['minmax','z-score','maxabs'][1])
        sample = np.transpose(sample)
        sample = torch.tensor(sample, dtype=torch.float32)     # 将样本转换为特定格式
        label = torch.tensor(label, dtype=torch.long)
        label = label.squeeze(1)                                # 将标签转换为只有batchsize这一个维度的数据结构，不然会报错
        return sample,label

##########################################################################
    def SelfBearing_loader(self):  # 专门针对行星齿轮故障信号数据集的样本加载函数
        DATA = np.load(self.dataset_base_path + '/自建轴承/' + self.dataset_choose + '.npz')
        sample = DATA['sample']
        label = DATA['label']

        # 数据筛选，筛选出部分特征的数据
        index_selected = (label[:, 0] < 5)  # 只取前5种故障，
        sample = sample[index_selected, :]
        label = label[index_selected, :]
        # 数据量的选择，
        sample = sample[self.no_start:self.no_end, :] * 1
        label = label[self.no_start:self.no_end, :]

        # 数据归一化
        sample = np.transpose(sample)
        sample, scaler = self.sample_scaler(sample, mode=['minmax', 'z-score', 'maxabs'][1])
        sample = np.transpose(sample)
        sample = torch.tensor(sample, dtype=torch.float32)  # 将样本转换为特定格式
        label = torch.tensor(label, dtype=torch.long)
        label = label.squeeze(1)  # 将标签转换为只有batchsize这一个维度的数据结构，不然会报错
        return sample, label

##########################################################################
    def IMSBearingFaultRaw_loader(self):        # 专门针对行星齿轮故障信号数据集的样本加载函数
        DATA = np.load('D:\迁移学习\迁移学习-自建\故障信号\datasets/IMSBearingDataSet-分故障整理/' + self.dataset_choose  + '.npz')
        sample = DATA['sample']
        label  = DATA['label']
        day   = DATA['day']
        # 数据筛选，筛选出部分特征的数据
        '''
        index_selected = (load[:,0]==1) & (load[:,1]<=5)
        sample = sample[index_selected]
        label  = label[index_selected]
        load   = load[index_selected] 
        '''
        # 数据量的选择，
        sample = sample[self.no_start:self.no_end,:]*1
        label  = label[self.no_start:self.no_end,:]
        day    = day[self.no_start:self.no_end,:] 
        # 数据归一化
        sample = np.transpose(sample)
        sample,scaler = self.sample_scaler(sample,mode=['minmax','z-score','maxabs'][1])
        sample = np.transpose(sample)
        sample = torch.tensor(sample, dtype=torch.float32)     # 将样本转换为特定格式
        label = torch.tensor(label, dtype=torch.long)
        label = label.squeeze(1)                                # 将标签转换为只有batchsize这一个维度的数据结构，不然会报错
        return sample,label       


##########################################################################  
    def ZouBearingFaultRaw_loader(self):        # 专门针对行星齿轮故障信号数据集的样本加载函数
        DATA = np.load('D:\迁移学习\迁移学习-自建\故障信号\datasets/ZouBearingFaultRaw-恒定工况-王鑫整理/' + self.dataset_choose  + '.npz')
        sample = DATA['sample']
        label  = DATA['label']
        load   = DATA['load']
        # 数据筛选，筛选出部分特征的数据
        index_selected = (label[:,0]<4)#&(load[:,0]==0)     # 只取前四种故障，0：normal， 1：inner rac， 2：ball， 3：outerrace-1   4：outerrace2
        sample = sample[index_selected]
        label  = label[index_selected]
        load   = load[index_selected] 
        # 数据量的选择，
        sample = sample[self.no_start:self.no_end,:]*1
        label  = label[self.no_start:self.no_end,:]
        load   = load[self.no_start:self.no_end,:] 
        # 数据归一化
        sample = np.transpose(sample)
        sample,scaler = self.sample_scaler(sample,mode=['minmax','z-score','maxabs'][1])
        sample = np.transpose(sample)
        sample = torch.tensor(sample, dtype=torch.float32)     # 将样本转换为特定格式
        label = torch.tensor(label, dtype=torch.long)
        label = label.squeeze(1)                                # 将标签转换为只有batchsize这一个维度的数据结构，不然会报错
        return sample,label       


    
##########################################################################  
    def PlanetGearFaultFea_loader(self):    # 专门针对行星齿轮故障特征数据集的样本构造与加载函数
        DATA = np.load('D:\迁移学习\迁移学习-自建\故障信号\datasets/PlanetGearFaultRaw-原始数据与特征数据/' + self.dataset_choose  + '.npz')
        sample = DATA['sample'][self.no_start:self.no_end,:]#[:,0:25]
        label  = DATA['label'][self.no_start:self.no_end,:]
        load   = DATA['load'][self.no_start:self.no_end,:] 
        
        # 数据筛选，筛选出部分特征的数据
        index_selected =  (load[:,0]<4)#&(load[:,0]==0)     # 只取前四种故障，0：normal， 1：inner rac， 2：ball， 3：outerrace-1   4：outerrace2
        sample = sample[index_selected,:]   
        label  = label[index_selected,:]
        load   = load[index_selected,:]        
        '''        
        if self.scaler=='no_scaler':    # 源域训练则不提供scaler
            sample,self.scaler = self.sample_scaler(sample,mode=['minmax','z-score','maxabs'][1])
        else:                           # 目标域训练使用源域的scaler
            sample = self.scaler.fit_transform(sample) 
        '''
        sample,self.scaler = self.sample_scaler(sample,mode=['minmax','z-score','maxabs'][1])
        sample = torch.tensor(sample, dtype=torch.float32)     # 将样本转换为特定格式
        label = torch.tensor(label, dtype=torch.long)
        label = label.squeeze(1)                                # 将标签转换为只有batchsize这一个维度的数据结构，不然会报错
        return sample, label        






    ## 样本归一化、标准化的函数
    def sample_scaler(self,sample,mode='minmax'):
        if mode == 'minmax':
            scaler = preprocessing.MinMaxScaler()
            sample_scaled = scaler.fit_transform(sample)
        if mode == 'maxabs':
            scaler = preprocessing.MaxAbsScaler()
            sample_scaled = scaler.fit_transform(sample)
        if mode == 'z-score':
            scaler = preprocessing.StandardScaler()
            sample_scaled = scaler.fit_transform(sample) 
        return sample_scaled,scaler            

