import numpy as np

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

def transform_to_woe(rawdata,woe_data):
    # i 为rawdata的列index，最后一列为target
    # j 为i 这个字段取到的值的index
    data_woe = np.zeros(np.shape(rawdata))
    for i in range(np.shape(rawdata)[1]-1):
        keys_val = list(woe_data[i].keys())
        woe_val = list(woe_data[i].values())
        for j in range(np.shape(woe_val)[0]):
            change_index = np.where(keys_val[j] == rawdata[:, i])[0].tolist()
            for k in change_index:
                data_woe[k,i] = woe_val[j]
    data_woe[:, -1] = rawdata[:, -1]
    return data_woe



use_data_comb = np.load(r'D:/rm/0926/use_data_comb_0926.npy')
woe_0926 = np.load(r'D:/rm/0926/woe_0926.npy')


use_data_woe=transform_to_woe(use_data_comb, woe_0926)
np.save( 'D:\\rm\\0926\\use_data_woe_0926', use_data_woe)





