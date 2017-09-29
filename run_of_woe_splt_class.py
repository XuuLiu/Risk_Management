import woe_splt_class as wsc
import numpy as np

def loadDataSet(fileName):  # 读取文件，txt文档
    fr = open(fileName, 'r', encoding='UTF-8')
    numFeat = len(fr.readline().split('\t'))  # 自动检测特征的数目
    dataMat = []
    fr = open(fileName, 'r', encoding='UTF-8')
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(curLine[i])
        dataMat.append(lineArr)
    return dataMat

#读数据
use_data_raw=np.array(loadDataSet('D:\\rm\\0926\\use_data_raw0926.txt'))
val_index_26new=np.array(loadDataSet(r'D:/rm/0926/val_index_26new.txt'))

#分箱，并把分箱结果存在路径
save_root='D:/rm/0926'
ws=wsc.group_by_woe(save_root)

group_data_str=ws.str_splt(use_data_raw[:,0:15],use_data_raw[:,-1],val_index_26new[0:15,:])
group_data_flt=ws.flt_splt(use_data_raw[:,15:26],use_data_raw[:,-1],val_index_26new[15:26,:])
#所有分箱好的数据
group_data_all=np.hstack((group_data_str,group_data_flt,use_data_raw[:,-1].reshape(np.shape(use_data_raw)[0],1)))
ws.ExportData(group_data_all, r'%s\group_data_all_0929.txt' % save_root)

#看看分箱好数据的iv和woe
iv=wsc.WOE(None)
woe_group, iv_group = iv.woe(group_data_all[:, 0:26], group_data_all[:, -1],event='1')

ws.ExportData(iv_group.reshape(np.shape(iv_group)[0],1), r'%s\iv_group_0929.txt' % save_root)

