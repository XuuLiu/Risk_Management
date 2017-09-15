# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:32:17 2017

@author: xu.liu
"""

import numpy as np
import random
from sklearn import preprocessing
import sys
sys.path.append('D:\\rm\\0913')
import information_value as iv
from prettytable import PrettyTable
random.seed(1000)

def loadDataSet(fileName):   #读取文件，txt文档
    fr = open(fileName,'r',encoding='UTF-8')
    numFeat = len(fr.readline().split('\t')) #自动检测特征的数目
    dataMat = []
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):      
            lineArr.append(curLine[i])
        dataMat.append(lineArr)
    return dataMat

def ExportData(data,fileName):#导出
    f=open(fileName,'w')
    for i in data:
        k=''.join([str(j) for j in i])
        f.write(k+"\n")
    f.close()   

def rd_pick_smp(importdata,totnum,num1,num0):
    '''importdata为原始输入数据集，list格式，最后一列表示为target，以yes和no的形式表示
        totnum为总共的样本数据量
        num1为样本数据中target=1的数据的量
        num0为样本数据中target=0的数据的量
        返回值为list
    '''
    smp_data=np.zeros((totnum,np.shape(importdata)[1]))
    smp_data = smp_data.astype(np.str)
    rd_data_0lt=[]#target为0即no的全部数据
    rd_data_1lt=[]#target为1即yes的全部数据
    for i in range(np.shape(importdata)[0]):
        if importdata[i][-1]=='yes':
            rd_data_1lt.append(importdata[i])
        elif importdata[i][-1]=='no':
            rd_data_0lt.append(importdata[i])
        
    smp_data_lt=[]#样本数据
    
    random.shuffle(rd_data_1lt)
    smp_data_1lt = random.sample(rd_data_1lt, num1)
    for i in range(np.shape(smp_data_1lt)[0]):
        smp_data_lt.append(smp_data_1lt[i])
        
    random.shuffle(rd_data_0lt)
    smp_data_0lt = random.sample(rd_data_0lt, num0)
    for i in range(np.shape(smp_data_0lt)[0]):
        smp_data_lt.append(smp_data_0lt[i])
         
    random.shuffle(smp_data_lt)
    for i in range(np.shape(smp_data_lt)[0]):
        if smp_data_lt[i][-1]=='yes':
            smp_data_lt[i][-1]=1
        elif smp_data_lt[i][-1]=='no':
            smp_data_lt[i][-1]=0
        
    return smp_data_lt

def get_val_name(val_index_file,index):
    '''通过变量index找到对应的变量名，val_index_file第一列为index，第二列为变量名'''
    for i in range(np.shape(val_index_file)[0]):
        if int(val_index_file[i][0])==index:
            return val_index_file[i][1]

def draw_tb_woe(woe_data,val_index):
    '''画woe表格'''
    tab = PrettyTable()
    # 设置表头
    tab.field_names = ["val_index","val_name", "val_value","woe"]
    # 表格内容插入
    for i in range(np.shape(woe_data)[0]):
        key=list(woe_data[i].keys())
        value=list(woe_data[i].values())
        for j in range(np.shape(key)[0]):
            tab.add_row([i,get_val_name(val_index,i),key[j],round(value[j],3)])
    tab_info = str(tab)
    print(tab_info)
    
def draw_tb_iv(iv_data,val_index):
    '''画iv表格'''
    tab = PrettyTable()
    # 设置表头
    tab.field_names = ["val_index", "val_name","iv"]
    # 表格内容插入
    for i in range(np.shape(iv_data)[0]):
        tab.add_row([i,get_val_name(val_index,i),round(iv_data[i],3)])
    tab_info = str(tab)
    space = 5
    print(tab_info)


            
#读取数据   
imp_datarote = 'D:\\rm\\0913\\raw0913.txt'
imp_data_lst=loadDataSet(imp_datarote)
#第一列序号为0的时候会不读，不知道问题在哪里，文件将index=0的放最后,第一列为index，第二列为变量名称
val_indexrt = 'D:\\rm\\0913\\val_index.txt'
val_index=loadDataSet(val_indexrt)
#取样本数据
smp_data=np.array(rd_pick_smp(imp_data_lst,4200,840,3360))

#计算woe和iv
woe=iv.WOE()
woe_all,iv_all=woe.woe(smp_data[:,0:np.shape(smp_data)[1]-1],smp_data[:,np.shape(smp_data)[1]-1],event='1')
#画表格
draw_tb_woe(woe_all,val_index)
draw_tb_iv(iv_all,val_index)

#将iv中小于0.03的变量删除，此处index为9，10，11，12，18，19，21

str_index=[0,	1,	2,	3,	6,	7,	8,	13,	14,	15,	16,	17,	20,	22,	26,	27,	33]#保留的离散型变量
flt_index=[4,	5,	23,	24,	25,	28,	29,	30,	31,	32,	34]#保留的连续型变量

use_data_raw=np.zeros((4200,np.shape(smp_data)[1]-7))
use_data_raw=use_data_raw.astype(np.str)
for i in range(np.shape(str_index)[0]):
    use_data_raw[:,i] = smp_data[:,str_index[i]]  
for i in range(np.shape(flt_index)[0]):
    use_data_raw[:,np.shape(str_index)[0]+i] =smp_data[:,flt_index[i]]
use_data_raw[:,-1]=smp_data[:,-1]       

#导出数据
ExportData(a,'D:\\rm\\0913\\20.txt')