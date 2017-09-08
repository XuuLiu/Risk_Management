# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

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

def ExportData(fileName):
    f=open(fileName,'w')
    for i in enew_m:
        k=','.join([str(j) for j in i])
        f.write(k+"\n")
    f.close()
           

enew_filename = 'D:\\rm\\0906\\enew.txt'
enew=np.array(loadDataSet(enew_filename))

city_filename='D:\\rm\\0906\\city_dic.txt'
city_dic=np.array(loadDataSet(city_filename))

enew_m=np.zeros(np.shape(enew))
enew_m = enew_m.astype(np.str)



for i in range(np.shape(enew)[0]):
    enew_m[i,:]=enew[i,:]
    for j in range(np.shape(city_dic)[0]):
        if enew[i,3]==city_dic[j,0]:
            enew_m[i,3]=city_dic[j,1]
        if enew[i,5]==city_dic[j,0]:
            enew_m[i,5]=city_dic[j,1]
        if enew[i,7]==city_dic[j,0]:
            enew_m[i,7]=city_dic[j,1]
        if enew[i,8]==city_dic[j,0]:
            enew_m[i,8]=city_dic[j,1]
        if enew[i,11]==city_dic[j,0]:
            enew_m[i,11]=city_dic[j,1]
        if enew[i,13]==city_dic[j,0]:
            enew_m[i,13]=city_dic[j,1]


ExportData( 'D:\\rm\\0906\\enew_modified.csv')

