# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:29:12 2017

@author: xu.liu
"""

from sklearn.linear_model import LogisticRegression 
import numpy as np
from sklearn import preprocessing
import random
from sklearn import metrics
from scipy import stats
import math
import matplotlib.pyplot as plt
from pylab import *

def loadDataSet(fileName):   #读取文件，txt文档
    fr = open(fileName,'r',encoding='UTF-8')
    numFeat = len(fr.readline().split('\t')) #自动检测特征的数目
    dataMat = []
    fr = open(fileName,'r',encoding='UTF-8')
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
        k='\t'.join([str(j) for j in i])
        f.write(k+"\n")
    f.close()

def vector(datastring):
    #字符型转化为因子
    le = preprocessing.LabelEncoder()
    datastringarr = np.array(datastring)
    labeldatain=[]
    for i in range(np.shape(datastringarr)[0]):#离散型变量数
        le.fit(datastringarr[i])
        labeldatain.append(le.transform(datastringarr[i]))
    labeldataT=np.array(labeldatain).T
    return labeldataT 

def inspectvector(string,vector,n):
#infostring为原始的离散型变量取值，infovector为因子表，n为要验证的变量在infostring第几列（python计数）
    attr= list(set(string[:,n]))
    vect=[]
    for i in range(np.shape(attr)[0]):
        a=[]
        for m in range(2):
            a.append("a")
        for j in range(np.shape(string[:,n])[0]):
            if attr[i]==string[j][n]:
                a[0]=attr[i]
                a[1]=vector[j,n]
        vect.append(a)
    return vect

def inspectvector_all(data_str,data_vector,val_index):
    # data_str是未替换为因子的数据集，data_vector为已替换为因子的数据集，方法为返回所有的因子与变量的对应字典
    vector_all=[]
    for i in range(np.shape(data_str)[1]-1):
        vector=inspectvector(data_str[:,0:28],data_vector,i)
        vector_one=[]
        for j in range(np.shape(vector)[0]):
            vector_one_row=[]
            vector_one_row.append(val_index[i][1])
            vector_one_row.append(vector[j][0])
            vector_one_row.append(vector[j][1])
            vector_one.append(list(vector_one_row))
        for k in range(np.shape(vector_one)[0]):
            vector_all.append(list(vector_one[k]))
    return vector_all


def value_times(data, value):
    # 用于统计value值在data（为一维数组）中出现的次数
    count = 0
    for i in range(np.shape(data)[0]):
        if data[i] == value:
            count += 1
    return count




use_data_comb = np.load(r'D:/rm/0926/use_data_comb_0926.npy')
#字符型转化为因子,连续的不变
use_data_comb_vector=vector(use_data_comb.T[0:15])
val_index_26new=loadDataSet(r'D:/rm/0926/val_index_26new.txt')
#查看因子代表的含义,14个离散的变量
vector_all=inspectvector_all(use_data_comb[:,0:15],use_data_comb_vector,val_index_26new[0:15])
       
ExportData(vector_all, r'D:\rm\0926\vector.txt')

#将转化为数字的变量转化为哑变量
enc = preprocessing.OneHotEncoder()
enc.fit(use_data_comb_vector)
use_data_comb_dump=list((enc.transform(use_data_comb_vector).toarray()).T)

a=[]
for i in range(np.shape(use_data_comb_dump)[0]-1):
    a.append((use_data_comb_dump)[i].tolist())
a.append(use_data_comb[:,-1].tolist())
model_data=np.array(a).T.astype(float)

ExportData(model_data,r'D:\rm\0926\model_data_40.txt')

#####################################################################################

inmodel_data = np.load(r'D:/rm/0926/use_data_woe_0926.npy')

model_data=hstack((inmodel_data[:,0:5],inmodel_data[:,6:9],inmodel_data[:,14:27]))

random.shuffle(model_data)

train_data=model_data[0:3360,:]
test_data=model_data[3360:4200,:]

ExportData(train_data,r'D:\rm\0926\train_data_20.txt')
ExportData(train_data,r'D:\rm\0926\train_data_20.txt')


classifier = LogisticRegression()
classifier.fit(train_data[:,12:14], train_data[:,-1])

#predic_y = classifier.predict(test_data[:,0:24])
predic_porb_y=classifier.predict_proba(test_data[:,12:14])
test_auc = metrics.roc_auc_score(test_data[:,-1],predic_porb_y[:, 1])
print(test_auc)

test_ks=stats.ks_2samp(predic_porb_y[:, 1],predic_porb_y[:, 0])[0]

ExportData(predic_porb_y,'D:\\rm\\0926\\predic_porb_y_0926.txt')

################################################################################

def draw_ks(test_data, predic_porb_y, group_contain):
    test_target = test_data[:, -1].reshape(840, 1)
    draw_ks = np.hstack((predic_porb_y, test_target))
    draw_ks_sort = draw_ks[np.lexsort(draw_ks[:, ::-1].T)]
    # 一共840行，每组20行，共42组
    # group_contain=20
    group_no = 0
    count_1 = []
    count_0 = []
    for i in range(math.ceil(np.shape(draw_ks_sort)[0] / group_contain)):
        group = draw_ks_sort[i + group_no * group_contain:i + (group_no + 1) * group_contain, :]
        count_1.append(sum(group[:, -1]))
        count_0.append(group_contain - sum(group[:, -1]))
        group_no += 1

    ###label为0的累积百分比
    add_0 = []
    add_0.append(count_0[0] / np.shape(draw_ks_sort)[0])
    for i in range(np.shape(count_0)[0] - 1):
        add_0.append(add_0[i] + count_0[i + 1] / value_times(draw_ks_sort[:, -1], 0))

    ###label为1的累积百分比
    add_1 = []
    add_1.append(count_1[0] / np.shape(draw_ks_sort)[0])
    for i in range(np.shape(count_1)[0] - 1):
        add_1.append(add_1[i] + count_1[i + 1] / value_times(draw_ks_sort[:, -1], 1))

    max_gap = [0, 0, 0, 0, 0]  # 差值、1的取值、0的取值、i、1比0
    for i in range(np.shape(add_0)[0]):
        if add_1[i] - add_0[i] > max_gap[0] and add_0[i] != 0:
            max_gap[0] = add_1[i] - add_0[i]
            max_gap[1] = add_1[i]
            max_gap[2] = add_0[i]
            max_gap[3] = i
            max_gap[4] = add_1[i] / add_0[i]

    mpl.rcParams['font.sans-serif'] = ['SimHei']

    names = range(math.ceil(np.shape(draw_ks_sort)[0] / group_contain))
    x = range(len(names))
    y0 = add_0
    y1 = add_1

    plt.plot(x, y0, marker='o', mec='r', mfc='w', label=u'0曲线图')
    plt.plot(x, y1, marker='*', ms=10, label=u'1曲线图')
    plt.legend()  # 让图例生效
    plt.xticks(x, names, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"分组")  # X轴标签
    plt.ylabel("累积比例")  # Y轴标签
    plt.title("KS")  # 标题
    plt.show()
    return max_gap


gaps=draw_ks(test_data,predic_porb_y,group_contain=20)


#系数
coef=classifier.coef_
intercept = classifier.intercept_

score=np.zeros((np.shape(test_data)[0],np.shape(test_data)[1]+2))
#测试数据评分表
for i in range(np.shape(test_data)[0]):
    for j in range(np.shape(test_data)[1]-2):
        score[i,j]=test_data[i,j]*coef[0,j]
    score[i,-3]=test_data[i,-1]
    score[i,-2]=predic_porb_y[i,1]
    score[i,-1]=576.41-16.67*(intercept+sum(score[i,0:26]))

ExportData(score,'D:\\rm\\0926\\score_0927.txt')




    
