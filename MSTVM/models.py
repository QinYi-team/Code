# 搭配模型
import torch.nn as nn
import torch
#import torchvision

import mmd
import backbone
from backbone import *
from config import CFG
import copy
DEVICE = CFG['DEVICE']
#DEVICE = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')


class Transfer_Net(nn.Module): 
    def __init__(self, CFG):     
        super(Transfer_Net, self).__init__()
        self.model = NetConv1d_MSTVM(CFG)
        self.DomClf_net = DomClf_FCL_5()
        # 加载FE_net在源域训练后的参数
        target_model_state = self.model.state_dict()         # 目标域模型的状态参数   
        source_pt_name = CFG['pt_name']      # 源域pt文件的名字
        source_pt = torch.load(source_pt_name + '.pt', map_location=DEVICE)              # 加载源域pt文件中的所有状态参数   
        source_model_state = source_pt['model']                     # 源域模型参数

        source_model_state_temp = {key:value   for key,value in source_model_state.items() if  'Clf' not in key}    # 只保留源域模型参数中含有“FE”字段的部分
        target_model_state.update(source_model_state_temp)          # 用源域已经训练好的底层，替换目标域的底层        
        self.model.load_state_dict(target_model_state)       # 将更新好的模型参数（源域的FE，重新初始化的Mid和Clf）


    # 前馈方法    
    def forward(self, data):
        clfout,feamap = self.model(data)     # midout用于输出到Clf中进行分类，feamap用于实现迁移mmd的损失计算
        return clfout, feamap       # clfout是最后一层输出额分类结果，feamap则是迁移特征 
        
        
        
        
        
