import torch
CFG = {
    'data_path': 'D:/data/Office31/Original_images/',
    'kwargs': {'num_workers': 5},
    'batch_size': 64,         # 训练集batchsize
    'batch_size_test': 64,    # 测试集batchsize
    'epoch': 50,              # 迭代次数
    'lr_step': 1,
    'gamma': 0.95,
    'lr': 0.001,           # 学习速率  learning rate
    'momentum': 0.8,       # 动量项
    'log_interval': 10, 
    'l2_decay': 0,
    'lambda': 0.02,
    'num_class': 4,        # 故障类别
    'DEVICE':torch.device('cuda'),  # 计算设备
    
}



