# -*- coding: utf-8 -*-


'''
在raw_data上选择性的筛选了27个变量，
不包括bankid_city和status_account
'''
import sys
sys.path.append(r'E:\pycharm\project1')
import math
import numpy as np
import information_value as iv
from prettytable import PrettyTable

'''变量分组函数'''

use_data_comb=np.zeros((4200,27))
use_data_comb=use_data_comb.astype(np.str)

val1_comb0=['14',	'16','61',	'62','75',	'96','BA','G3','G7','HW','I9','JA','KJ','O4','Y1','ZG']
val1_comb1=['0',	'I8']
val1_comb2=['65',	'51',	'1','CD',	'1002']
val1_comb3=['20',	'WF']
val4_comb0=['1B',	'1F']
val4_comb1=['1H',	'1G',	'1C',	'1A',	'1D',	'1E']
val4_comb2=['null']
val9_comb0=['null',	'青海']
val9_comb1=['上海',	'北京',	'江苏',	'辽宁',	'黑龙江',	'陕西',	'安徽',	'天津',	'广东',	'山东']
val9_comb2=['四川',	'湖北',	'宁夏',	'江西',	'河南',	'山西',	'重庆',	'河北',	'浙江']
val9_comb3=[	'吉林',	'福建',	'新疆',	'广西',	'海南',	'内蒙古',	'湖南']
val9_comb4=['甘肃',	'云南',	'西藏',	'贵州']

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
for i in range(np.shape(use_data_raw)[0]):
    use_data_comb[i,2]=use_data_raw[i,2]
    use_data_comb[i,3]=use_data_raw[i,3]
    if use_data_raw[i,4] in val4_comb0:
         use_data_comb[i,4]='val4_comb0'
    elif use_data_raw[i,4] in val4_comb1:
        use_data_comb[i,4]='val4_comb1'
    elif use_data_raw[i,4] in val4_comb2:
        use_data_comb[i,4]='val4_comb2'

for i in range(np.shape(use_data_raw)[0]):
    use_data_comb[i,5]=use_data_raw[i,5]
    use_data_comb[i,6]=use_data_raw[i,6]
    use_data_comb[i,7]=use_data_raw[i,7]
    use_data_comb[i,8]=use_data_raw[i,8]

    if use_data_raw[i,9] in val9_comb0:
        use_data_comb[i,9]='val9_comb0'
    elif use_data_raw[i,9] in val9_comb1:
        use_data_comb[i,9]='val9_comb1'
    elif use_data_raw[i,9] in val9_comb2:
        use_data_comb[i,9]='val9_comb2'
    elif use_data_raw[i,9] in val9_comb3:
        use_data_comb[i,9]='val9_comb3'
    elif use_data_raw[i,9] in val9_comb4:
        use_data_comb[i,9]='val9_comb4'

    use_data_comb[i,10]=use_data_raw[i,10]
    use_data_comb[i,11]=use_data_raw[i,11]
    use_data_comb[i,12]=use_data_raw[i,12]
    use_data_comb[i,13]=use_data_raw[i,13]
    use_data_comb[i,14]=use_data_raw[i,14]

#变量hit_time
for i in range(np.shape(use_data_raw)[0]):
    if float(use_data_raw[i,15])>0 and  float(use_data_raw[i,15])<=44:
        use_data_comb[i,15]='(0,44]'
    elif float(use_data_raw[i,15])>44 and  float(use_data_raw[i,15])<=56:
        use_data_comb[i,15]='(44,56]'
    elif float(use_data_raw[i,15])>56 and  float(use_data_raw[i,15])<=93:
        use_data_comb[i,15]='(56,93]'
    elif float(use_data_raw[i,15])>93 and  float(use_data_raw[i,15])<=111:
        use_data_comb[i,15]='(93,111]'
    elif float(use_data_raw[i,15])>111 and  float(use_data_raw[i,15])<=131:
        use_data_comb[i,15]='(111,131]'
    elif float(use_data_raw[i,15])>131 and  float(use_data_raw[i,15])<=179:
        use_data_comb[i,15]='(131,179]'
    elif float(use_data_raw[i,15])>179 and  float(use_data_raw[i,15])<=232:
        use_data_comb[i,15]='(179,232]'
    elif float(use_data_raw[i,15])>232 and  float(use_data_raw[i,15])<=645:
        use_data_comb[i,15]='(232,645]'
    elif float(use_data_raw[i,15])>645:
        use_data_comb[i,15]='(645,+inf)'
#变量pay_amount        
for i in range(np.shape(use_data_raw)[0]):
    pay_amount=float(use_data_raw[i,16])/100
    if float(pay_amount)>0 and  float(pay_amount)<=4.99:
        use_data_comb[i,16]='(0,499]'
    elif float(pay_amount)>4.99 and  float(pay_amount)<=49.99:
        use_data_comb[i,16]='(499,4999]'
    elif float(pay_amount)>49.99 and  float(pay_amount)<=79.99:
        use_data_comb[i,16]='(4999,7999]'
    elif float(pay_amount)>79.99 and  float(pay_amount)<=139.99:
        use_data_comb[i,16]='(7999,13999]'
    elif float(pay_amount)>139.99 and  float(pay_amount)<=199.99:
        use_data_comb[i,16]='(13999,19999]'
    elif float(pay_amount)>199.99 and  float(pay_amount)<=349.99:
        use_data_comb[i,16]='(19999,34999]'
    elif float(pay_amount)>349.99:
        use_data_comb[i,16]='(34999,+inf)'

#变量age
for i in range(np.shape(use_data_raw)[0]):
    if float(use_data_raw[i,17])>0 and  float(use_data_raw[i,17])<=20:
        use_data_comb[i,17]='(0,20]'
    elif float(use_data_raw[i,17])>20  and  float(use_data_raw[i,17])<=28:
        use_data_comb[i,17]='(20,28]'
    elif float(use_data_raw[i,17])>28  and  float(use_data_raw[i,17])<=36:
        use_data_comb[i,17]='(28,36]'
    elif float(use_data_raw[i,17])>36  and  float(use_data_raw[i,17])<=59:
        use_data_comb[i,17]='(36,59]'
    elif float(use_data_raw[i,17])>59:
        use_data_comb[i,17]='(59,+inf)'
#变量sign_card
for i in range(np.shape(use_data_raw)[0]):
    mins=math.ceil(float(use_data_raw[i,18])/60)
    if mins>0 and mins<=8:
        use_data_comb[i,18]='(0,8]'
    elif mins>8 and mins<=260:
        use_data_comb[i,18]='(8,260]'
    elif mins>260 and mins<=150000:
        use_data_comb[i,18]='(260,150000]'
    elif mins>150000 and mins<=250423:
        use_data_comb[i,18]='(150000,250423]'
    elif mins>250423 and mins<=264340:
        use_data_comb[i,18]='(250423,264340]'
    elif mins>264340 and mins<=460709:
        use_data_comb[i,18]='(264340,460709]'
    elif mins>460709:
        use_data_comb[i,18]='(460709,+inf)'

#card_txn      
for i in range(np.shape(use_data_raw)[0]):
    mins=math.ceil(float(use_data_raw[i,19])/60)
    if mins>0 and mins<=2:
        use_data_comb[i,19]='(0,2]'
    elif mins>2 and mins<=29:
        use_data_comb[i,19]='(2,29]'
    elif mins>29 and mins<=4320:#3天
        use_data_comb[i,19]='(29,4320]'
    elif mins>4320 and mins<=57600:#40天
        use_data_comb[i,19]='(4320,57600]'
    elif mins>57600 and mins<=266400:#185天
        use_data_comb[i,19]='(57600,266400]'
    elif mins>266400 and mins<=280800:
        use_data_comb[i,19]='(266400,280800]'
    elif mins>280800:
        use_data_comb[i,19]='(280800,+inf)'

#card_upd        
for i in range(np.shape(use_data_raw)[0]):
    mins=math.ceil(float(use_data_raw[i,20])/60)
    if mins>=0 and mins<=2:
        use_data_comb[i,20]='[0,2]'
    elif mins>2 and mins<=1440:#1天
        use_data_comb[i,20]='(2,1440]'
    elif mins>1440 and mins<=10080:#7天
        use_data_comb[i,20]='(1440,10080]'
    elif mins>10080 and mins<=525600:#1年
        use_data_comb[i,20]='(10080,525600]'
    elif mins>525600:
        use_data_comb[i,20]='(525600,+inf)'
#authid_card         
for i in range(np.shape(use_data_raw)[0]):
    mins=math.ceil(float(use_data_raw[i,21])/60)
    if mins<=0:
        use_data_comb[i,21]='(-inf,0]'
    elif mins>0 and mins<=4320:#3天
        use_data_comb[i,21]='(0,4320]'
    elif mins>4320 and mins<=28800:#20天
        use_data_comb[i,21]='(4320,28800]'
    elif mins>28800:
        use_data_comb[i,21]='(28800,+inf)'


for i in range(np.shape(use_data_raw)[0]):
    if float(use_data_raw[i,22])==0:
        use_data_comb[i,22]='0'
    elif float(use_data_raw[i,22])>0:
        use_data_comb[i,22]='(0,+inf)'

    if float(use_data_raw[i,23])==0:
        use_data_comb[i,23]='0'
    elif float(use_data_raw[i,23])>0:
        use_data_comb[i,23]='(0,+inf)'

    if float(use_data_raw[i,24])==0:
        use_data_comb[i,24]='0'
    elif float(use_data_raw[i,24])==1:
        use_data_comb[i,24]='1'
    elif float(use_data_raw[i,24])==2:
        use_data_comb[i,24]='2'
    elif float(use_data_raw[i,24])==3:
        use_data_comb[i,24]='3'
    elif float(use_data_raw[i,24])>3:
        use_data_comb[i,24]='(3,+inf)'

#txn_time_hour
for i in range(np.shape(use_data_raw)[0]):
    if float(use_data_raw[i,25])>=0 and float(use_data_raw[i,25])<=7:
        use_data_comb[i,25]='[0,7]'
    elif float(use_data_raw[i,25])>7 and float(use_data_raw[i,25])<=9:
        use_data_comb[i,25] ='(7,9]'
    elif float(use_data_raw[i,25])>9 and float(use_data_raw[i,25])<=17:
        use_data_comb[i,25]='(9,17]'
    elif float(use_data_raw[i,25])>17 and float(use_data_raw[i,25])<=22:
        use_data_comb[i,25]='(17,22]'
    elif float(use_data_raw[i,25])>22:
        use_data_comb[i,25]='(22,+inf)'
#加上目标变量
for i in range(np.shape(use_data_raw)[0]):
    use_data_comb[i,-1]=use_data_raw[i,-1]

#获取分组后数据集  

####################################################

def Table(data,iv_raw,val_index):
    woe=iv.WOE()   
    woecomb,ivcomb=woe.woe(data[:,0:np.shape(data)[1]-1],data[:,np.shape(data)[1]-1],event='1')
    ivtable=draw_tb(woecomb,iv_raw,ivcomb,val_index,data)
    return ivtable, ivcomb,woecomb

iv_0926,iv_c0926,woe_0926=Table(use_data_comb,iv_all,val_index_26new)


ExportData(use_data_comb,'D:\\rm\\0926\\use_data_comb_str.txt')






np.save( 'D:\\rm\\0926\\use_data_comb_0926', use_data_comb)
np.save( 'D:\\rm\\0926\\woe_0926', np.array(woe_0926))



