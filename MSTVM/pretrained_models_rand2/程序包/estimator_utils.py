from __future__ import print_function          # 调用Python未来的输出函数print
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import torch


def plot_curve(x,y,path_name,xlabel='Epoch',ylabel='Loss',title='Loss',legend=['Training_Loss','Testing_Loss']):
    fig = plt.figure()
    colors = ['r','g','r','g']
    linestyles = ['-','-','--','--']
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    for i,y_ in enumerate(y):
        plt.plot(x,y_,color=colors[i],linestyle=linestyles[i],  linewidth=2,label=legend[i])
    plt.legend(fontsize=18)
    plt.ylabel(ylabel,fontsize=18)    
    plt.xlabel(xlabel,fontsize=18) 
    plt.title(path_name,fontsize=8) 
    plt.tick_params(labelsize=15)

    plt.savefig(path_name, dpi=400) 
    plt.close()     
    


def plot_distribution_tsne(x,y,path_name):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    x_tsne = tsne.fit_transform(x)
    
    print("Org data dimension is {}\
          Embedded data dimension is {}".format(x.shape[-1], x_tsne.shape[-1]))
    
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)  # 归一化
    marks = ['ro','b>','k+','g*']
    colors = ['r','b','k','g','y','m','c','yellow']
    colors = ['r','brown','coral','maroon','greenyellow','lawngreen','g','darkgreen']
    colors = ['r','darkred','orangered','peru','gold',  'g','lime','b','olive','teal']    
    y = y.reshape([len(y),1])
    plt.figure(figsize=(8, 8))
    for i in range(x_norm.shape[0]):    
        plt.text(x_norm[i, 0], x_norm[i, 1], str(int(y[i,0])), color=colors[int(y[i,0])],  # 用于设置文字说明，在途中写文字
                 fontdict={'weight': 'bold', 'size': 25})
        #plt.plot(X_norm[i, 0], X_norm[i, 1], mark[int(y[i,0])])      # 用于设置文字说明，在途中写文字
    #plt.xticks([])
    #plt.yticks([])
    plt.axis('off')
    plt.savefig(path_name, dpi=400) 
    plt.close()         
    
    


    
    
    
    