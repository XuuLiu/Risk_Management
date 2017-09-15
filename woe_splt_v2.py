# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:20:06 2017

@author: xu.liu
"""

'''以下为根据woe来分组、分段,使用初始数据集为data_random_splt.py生成的use_data_raw'''
import numpy as np
import sys
sys.path.append('D:\\rm\\0913')
import information_value as iv
from prettytable import PrettyTable

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

def get_val_name(val_index_file,index):
    '''通过变量index找到对应的变量名，val_index_file第一列为index，第二列为变量名'''
    for i in range(np.shape(val_index_file)[0]):
        if int(val_index_file[i][0])==index:
            return val_index_file[i][1]  
        
def count_x_invol(data,x,volnum): 
    '''计数x在第volnum列中出现的百分比'''
    a=list(data[:,volnum])
    c=a.count(x)/np.shape(data)[0]
    return c        
        
def draw_tb(woe_data,iv_data_raw,iv_data_now,val_index,data_all):
    '''画woe和iv的总结表格
    data_all为用于计算woe和iv值的数据集'''
    tab = PrettyTable()
    # 设置表头
    tab.field_names = ["val_index","val_name","IV_raw", "IV_now","val_value","woe","count%"]
    # 表格内容插入
    for i in range(np.shape(woe_data)[0]):
        key=list(woe_data[i].keys())
        value=list(woe_data[i].values())
        for j in range(np.shape(key)[0]):
            tab.add_row([i,get_val_name(val_index,i),iv_data_raw[i],iv_data_now[i],key[j],round(value[j],3),count_x_invol(data_all,key[j],i)])
    tab_info = str(tab)
    print(tab_info)
    

use_data_comb=np.zeros(np.shape(use_data_raw))
use_data_comb=use_data_comb.astype(np.str)

val1_comb0=['14',	'16','61',	'62','75',	'96','BA','G3','G7','HW','I9','JA','KJ','O4','Y1','ZG']
val1_comb1=['0',	'I8']
val1_comb2=['65',	'51',	'1','CD',	'1002']
val1_comb3=['20',	'WF']
val5_comb0=['1B',	'1F']
val5_comb1=['1H',	'1G',	'1C',	'1A',	'1D',	'1E']
val5_comb2=['null']
val10_comb0=['null',	'青海']
val10_comb1=['上海',	'北京',	'江苏',	'辽宁',	'黑龙江',	'陕西',	'安徽',	'天津',	'广东',	'山东']
val10_comb2=['四川',	'湖北',	'宁夏',	'江西',	'河南',	'山西',	'重庆',	'河北',	'浙江']
val10_comb3=[	'吉林',	'福建',	'新疆',	'广西',	'海南',	'内蒙古',	'湖南']
val10_comb4=['甘肃',	'云南',	'西藏',	'贵州']


for i in range(np.shape(use_data_raw)[0]):
    use_data_comb[i,0]=use_data_raw[i,0]
    if use_data_raw[i,1] in val1_comb0:
        use_data_comb[i,1]='val1_comb0'
    elif use_data_raw[i,1] in val1_comb1:
        use_data_comb[i,1]='val1_comb1'
    elif use_data_raw[i,1] in val1_comb2:
        use_data_comb[i,1]='val1_comb2'
    elif use_data_raw[i,1] in val1_comb3:
        use_data_comb[i,1]='val1_comb3'
    use_data_comb[i,2]=use_data_raw[i,2]
    use_data_comb[i,3]=use_data_raw[i,3]
    use_data_comb[i,4]=use_data_raw[i,4]
    if use_data_raw[i,5] in val5_comb0:
        use_data_comb[i,5]='val5_comb0'
    elif use_data_raw[i,5] in val5_comb1:
        use_data_comb[i,5]='val5_comb1'
    elif use_data_raw[i,5] in val5_comb2:
        use_data_comb[i,5]='val5_comb2'
    use_data_comb[i,6]=use_data_raw[i,6]
    use_data_comb[i,7]=use_data_raw[i,7]
    use_data_comb[i,8]=use_data_raw[i,8]
    use_data_comb[i,9]=use_data_raw[i,9]
    if use_data_raw[i,10] in val10_comb0:
        use_data_comb[i,10]='val10_comb0'
    elif use_data_raw[i,10] in val10_comb1:
        use_data_comb[i,10]='val10_comb1'
    elif use_data_raw[i,10] in val10_comb2:
        use_data_comb[i,10]='val10_comb2'
    elif use_data_raw[i,10] in val10_comb3:
        use_data_comb[i,10]='val10_comb3'
    elif use_data_raw[i,10] in val10_comb4:
        use_data_comb[i,10]='val10_comb4'

    use_data_comb[i,11]=use_data_raw[i,11]
    use_data_comb[i,12]=use_data_raw[i,12]
    use_data_comb[i,13]=use_data_raw[i,13]
    use_data_comb[i,14]=use_data_raw[i,14]
    use_data_comb[i,15]=use_data_raw[i,15]
    use_data_comb[i,16]=use_data_raw[i,16]
    
    if float(use_data_raw[i,17])>0 and  float(use_data_raw[i,17])<=44:
        use_data_comb[i,17]='(0,44]'
    elif float(use_data_raw[i,17])>44 and  float(use_data_raw[i,17])<=66:
        use_data_comb[i,17]='(44,66]'
    elif float(use_data_raw[i,17])>66 and  float(use_data_raw[i,17])<=78:
        use_data_comb[i,17]='(66,78]'
    elif float(use_data_raw[i,17])>78 and  float(use_data_raw[i,17])<=94:
        use_data_comb[i,17]='(78,94]'
    elif float(use_data_raw[i,17])>94 and  float(use_data_raw[i,17])<=111:
        use_data_comb[i,17]='(94,111]'
    elif float(use_data_raw[i,17])>111 and  float(use_data_raw[i,17])<=131:
        use_data_comb[i,17]='(111,131]'
    elif float(use_data_raw[i,17])>131 and  float(use_data_raw[i,17])<=179:
        use_data_comb[i,17]='(131,179]'
    elif float(use_data_raw[i,17])>179 and  float(use_data_raw[i,17])<=232:
        use_data_comb[i,17]='(179,232]'
    elif float(use_data_raw[i,17])>232 and  float(use_data_raw[i,17])<=395:
        use_data_comb[i,17]='(232,395]'
    elif float(use_data_raw[i,17])>395:
        use_data_comb[i,17]='(395,+inf)'
        
    if float(use_data_raw[i,18])>0 and  float(use_data_raw[i,18])<=150:
        use_data_comb[i,18]='(0,150]'
    elif float(use_data_raw[i,18])>150 and  float(use_data_raw[i,18])<=500:
        use_data_comb[i,18]='(150,500]'
    elif float(use_data_raw[i,18])>150 and  float(use_data_raw[i,18])<=500:
        use_data_comb[i,18]='(150,500]'
    elif float(use_data_raw[i,18])>500 and  float(use_data_raw[i,18])<=1450:
        use_data_comb[i,18]='(500,1450]'
    elif float(use_data_raw[i,18])>1450 and  float(use_data_raw[i,18])<=2500:
        use_data_comb[i,18]='(1450,2500]'
    elif float(use_data_raw[i,18])>2500 and  float(use_data_raw[i,18])<=4900:
        use_data_comb[i,18]='(2500,4900]'
    elif float(use_data_raw[i,18])>4900 and  float(use_data_raw[i,18])<=10005:
        use_data_comb[i,18]='(4900,10005]'
    elif float(use_data_raw[i,18])>10005 and  float(use_data_raw[i,18])<=20000:
        use_data_comb[i,18]='(10005,20000]'
    elif float(use_data_raw[i,18])>20000:
        use_data_comb[i,18]='(20000,+inf)'

    if float(use_data_raw[i,19])>0 and  float(use_data_raw[i,19])<=20:
        use_data_comb[i,19]='(0,20]'
    elif float(use_data_raw[i,19])>20  and  float(use_data_raw[i,19])<=28:
        use_data_comb[i,19]='(20,28]'
    elif float(use_data_raw[i,19])>28  and  float(use_data_raw[i,19])<=36:
        use_data_comb[i,19]='(28,36]'
    elif float(use_data_raw[i,19])>36  and  float(use_data_raw[i,19])<=41:
        use_data_comb[i,19]='(36,41]'
    elif float(use_data_raw[i,19])>41 and  float(use_data_raw[i,19])<=50:
        use_data_comb[i,19]='(41,50]'
    elif float(use_data_raw[i,19])>50:
        use_data_comb[i,19]='(50,+inf)'
        
    if float(use_data_raw[i,20])>0 and float(use_data_raw[i,20])<=251:
        use_data_comb[i,20]='(0,251]'
    elif float(use_data_raw[i,20])>251 and float(use_data_raw[i,20])<=359:
        use_data_comb[i,20]='(251,359]'                    
    elif float(use_data_raw[i,20])>359 and float(use_data_raw[i,20])<=13566:
        use_data_comb[i,20]='(359,13566]'            
    elif float(use_data_raw[i,20])>13566 and float(use_data_raw[i,20])<=5224073:
        use_data_comb[i,20]='(13566,5224073]'           
    elif float(use_data_raw[i,20])>5224073 and float(use_data_raw[i,20])<=15025371:
        use_data_comb[i,20]='(5224073,15363357]'        
    elif float(use_data_raw[i,20])>15025371 and float(use_data_raw[i,20])<=15862388:
        use_data_comb[i,20]='(15363357,15862388]'
    elif float(use_data_raw[i,20])>15862388 and float(use_data_raw[i,20])<=27642526:
        use_data_comb[i,20]='(15862388,27642526]'
    elif float(use_data_raw[i,20])>27642526:
        use_data_comb[i,20]='(27642526,+inf)'
        
        
    if float(use_data_raw[i,21])>0 and float(use_data_raw[i,21])<=199:
        use_data_comb[i,21]='(0,199]'          
    elif float(use_data_raw[i,21])>199 and float(use_data_raw[i,21])<=330181:
        use_data_comb[i,21]='(199,330181]'                    
    elif float(use_data_raw[i,21])>330181 and float(use_data_raw[i,21])<=3369897:
        use_data_comb[i,21]='(330181,3369897]' 
    elif float(use_data_raw[i,21])>3369897 and float(use_data_raw[i,21])<=16407431:
        use_data_comb[i,21]='(3369897,16407431]'    
    elif float(use_data_raw[i,21])>16407431 and float(use_data_raw[i,21])<=16796511:
        use_data_comb[i,21]='(16407431,16796511]'
    elif float(use_data_raw[i,21])>16796511:
        use_data_comb[i,21]='(16796511,+inf)'

    if float(use_data_raw[i,22])>=0 and float(use_data_raw[i,22])<=24:
        use_data_comb[i,22]='[0,24]'
    elif float(use_data_raw[i,22])>24 and float(use_data_raw[i,22])<=64:
        use_data_comb[i,22]='(24,64]'
    elif float(use_data_raw[i,22])>64 and float(use_data_raw[i,22])<=1004:
        use_data_comb[i,22]='(64,1004]'                       
    elif float(use_data_raw[i,22])>1004 and float(use_data_raw[i,22])<=4828:
        use_data_comb[i,22]='(1004,4828]'            
    elif float(use_data_raw[i,22])>4828 and float(use_data_raw[i,22])<=16101765:
        use_data_comb[i,22]='(4828,16101765]'                 
    elif float(use_data_raw[i,22])>16101765:
        use_data_comb[i,22]='(16101765,+inf)'
   
    if float(use_data_raw[i,23])<=0:
        use_data_comb[i,23]='(-inf,0]'
    elif float(use_data_raw[i,23])>0 and float(use_data_raw[i,23])<=1000000:
        use_data_comb[i,23]='(0,1000000]'
    elif float(use_data_raw[i,23])>1000000:
        use_data_comb[i,23]='(1000000,+inf)'



    if float(use_data_raw[i,24])==0:
        use_data_comb[i,24]='0'
    elif float(use_data_raw[i,23])>0:
        use_data_comb[i,24]='(0,+inf)'
    
    if float(use_data_raw[i,25])==0:
        use_data_comb[i,25]='0'
    elif float(use_data_raw[i,25])>0:
        use_data_comb[i,25]='(0,+inf)'
        
    if float(use_data_raw[i,26])==0:
        use_data_comb[i,26]='0'
    elif float(use_data_raw[i,26])==1:
        use_data_comb[i,26]='1'
    elif float(use_data_raw[i,26])==2:
        use_data_comb[i,26]='2'
    elif float(use_data_raw[i,26])==3:
        use_data_comb[i,26]='3'
    elif float(use_data_raw[i,26])>3:
        use_data_comb[i,26]='(3,+inf)'  
        
    if float(use_data_raw[i,27])>=0 and float(use_data_raw[i,27])<=10:
        use_data_comb[i,27]='[0,10]'         
    elif float(use_data_raw[i,27])>10 and float(use_data_raw[i,27])<=15:
        use_data_comb[i,27]='(10,15]'
    elif float(use_data_raw[i,27])>15 and float(use_data_raw[i,27])<=17:
        use_data_comb[i,27]='(15,17]'
    elif float(use_data_raw[i,27])>17:
        use_data_comb[i,27]='(17,+inf)'  

    use_data_comb[i,-1]=use_data_raw[i,-1]
        
      
woe=iv.WOE()     
woe_comb,iv_comb=woe.woe(use_data_comb[:,0:np.shape(use_data_comb)[1]-1],use_data_comb[:,np.shape(use_data_comb)[1]-1],event='1')

#第一列序号为0的时候会不读，不知道问题在哪里，文件将index=0的放最后,第一列为index，第二列为变量名称
val_indexrt = 'D:\\rm\\0913\\val_index_28new.txt'
val_index_28new=loadDataSet(val_indexrt)

draw_tb(woe_comb,iv_all,iv_comb,val_index_28new,use_data_comb)