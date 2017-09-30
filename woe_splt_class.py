import numpy as np
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target
import scipy.cluster.hierarchy as sch

class WOE():
    def __init__(self,parent):
        self.parent = parent
        self._WOE_MIN = -20
        self._WOE_MAX = 20


    def woe(self, X, y, event=1):
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        self.check_target_binary(y)
        X1 = self.feature_discretion(X)

        res_woe = []
        res_iv = []
        for i in range(0, X1.shape[-1]):
            x = X1[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

    def woe_single_x(self, x, y, event=1):
        '''
        calculate woe and information for a single feature
        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        self.check_target_binary(y)

        event_total, non_event_total = self.count_binary(y, event=event)
        x_labels = np.unique(x)
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        '''
        replace the explanatory feature categories with its woe value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
        :return: the new numpy array in which woe values filled
        '''
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
            idx += 1

        return res

    def combined_iv(self, X, y, masks, event=1):
        '''
        calcute the information vlaue of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        '''
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv

    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    def check_target_binary(self, y):
        '''
        check if the target variable is binary, raise error if not.
        :param y:
        :return:
        '''
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')

    def feature_discretion(self, X):
        '''
        Discrete the continuous features of input data X, and keep other features unchanged.
        :param X : numpy array
        :return: the numpy array in which all continuous features are discreted
        '''
        temp = []
        for i in range(0, X.shape[-1]):
            x = X[:, i]
            x_type = type_of_target(x)
            if x_type == 'continuous':
                x1 = self.discrete(x)
                temp.append(x1)
            else:
                temp.append(x)
        return np.array(temp).T

    def discrete(self, x):
        '''
        Discrete the input 1-D numpy array using 5 equal percentiles
        :param x: 1-D numpy array
        :return: discreted 1-D numpy array
        '''
        res = np.array([0] * x.shape[-1], dtype=int)
        for i in range(5):
            point1 = stats.scoreatpercentile(x, i * 20)
            point2 = stats.scoreatpercentile(x, (i + 1) * 20)
            x1 = x[np.where((x >= point1) & (x <= point2))]
            mask = np.in1d(x, x1)
            res[mask] = (i + 1)
        return res

    @property
    def WOE_MIN(self):
        return self._WOE_MIN
    @WOE_MIN.setter
    def WOE_MIN(self, woe_min):
        self._WOE_MIN = woe_min
    @property
    def WOE_MAX(self):
        return self._WOE_MAX
    @WOE_MAX.setter
    def WOE_MAX(self, woe_max):
        self._WOE_MAX = woe_max



class group_by_woe():
    def __init__(self,save_root):
        self.save_root =save_root


    def str_splt(self, use_data_raw, use_data_target, val_index):

        '''
        This is used to group discrete variable
        
        use_data_raw为使用的未分箱的数据 are raw data that need to be grouped.

        1. 如果变量取值有大于等于5种才考虑分箱
            Only if the raw discrete variable has more than 5 values that will be regrouped.
        2. 离散数据每个取值之间的距离默认为单位距离
            Every value of discrete variable has unit distance.
        3. 分类后的IV值/原本的IV值要>=0.8
            IV before grouped/IV after grouped >=0.8
        4. 聚类采用系统聚类，且最大类别数为8
            Here use hierarchical clustering to group discrete variable values, and the maximum of group number is 8
        5. 因为我懒，所以采用先分类再看IV值是否满足条件3，如果不满足，则返回空，这样一点也不鲁棒
            Here check IV after cluster, if IV not satisfy the rule 3, the function will return []'''

        # find discrete variable that need to be grouped
        str_no=np.shape(use_data_raw)[1]

        Woe=WOE(None)
        woe_raw, iv_raw = Woe.woe(X=use_data_raw.reshape(np.shape(use_data_raw)[0],str_no),y=use_data_target, event='1')

        need_group = []
        for i in range(str_no):
            if len(woe_raw[i]) >= 5:
                need_group.append(i)

        # If there happens grouping, the result will be stored is the use_data_vector, or keep its raw value
        #若有分箱则改vector那个表，没有就保留原值
        use_data_vector = np.zeros(np.shape(use_data_raw))
        use_data_vector = use_data_vector.astype(np.str)
        use_data_vector = use_data_raw.copy()

        for i in need_group:
            print('Im working hard on grouping the %dth variance, may cost a little bit long time, plz wait :)' % i)
            woe = []
            woe.append(list(woe_raw[i].keys()))
            woe.append(list(woe_raw[i].values()))
            woe = np.array(woe).T

            # 生成点与点之间的距离矩阵,这里用的欧氏距离:
            # Generate the distance matrix of every value's WOE, here use euclidean distance
            disMat = sch.distance.pdist(woe[:, 1].reshape(np.shape(woe)[0], 1), 'euclidean')
            # 进行层次聚类:
            # hierarchical clustering
            Z = sch.linkage(disMat, method='average')
            # 将层级聚类结果以树状图表示出来
            # P=sch.dendrogram(Z)
            # 根据linkage matrix Z得到聚类结果:
            # Return clustering result according to linkage matrix Z
            cluster = sch.fcluster(Z, t=1, criterion='inconsistent')
            plus = 0
            while max(cluster) > 7:  
                cluster = sch.fcluster(Z, t=1 + plus, criterion='inconsistent')
                plus += 0.001
                if max(cluster) < 7 and max(cluster) > 2:
                    break

            # dictionary of grouping reuslt
            dic = np.hstack((woe, cluster.reshape(np.shape(cluster)[0], 1)))

            # replace raw data by its grouping result
            for j in range(np.shape(use_data_raw)[0]):
                for k in range(np.shape(dic)[0]):
                    if use_data_raw[j, i] == dic[k, 0]:
                        use_data_vector[j, i] = dic[k, 2]
                        
            # calculate woe and iv of grouped data
            Woe = WOE(None)
            woe_group, iv_group = Woe.woe(use_data_vector[:, i].reshape(np.shape(use_data_vector)[0], 1),use_data_target, event='1')
            
            # generate the formated dictionary for output
            out_dic = self.gener_dic(iv_group, woe_group, dic, iv_raw, i)
            if len(out_dic) > 0:
                self.ExportData(out_dic, r'%s\dic_of_%d%s.txt' % (self.save_root, i, val_index[i, 1]))
            else:
                self.ExportData('分组后iv缺失严重，请手动检查', r'%s\dic_of_%d%s.txt' % (save_root, i, val_index[i, 1]))
        return use_data_vector

    def flt_splt(self,use_data_raw, use_data_target,val_index):
        '''
        This is used to group continuous variable
        '''
        for i in range(np.shape(use_data_raw)[1] ):
            print('Im working hard on grouping the %dth variance, may cost a little bit long time, plz wait :)' % i)
            group_data, woe_splt = self.flt_splt_single(i, use_data_raw,use_data_target)
            use_data_raw[:, i] = group_data
            out_dic = woe_splt.copy().astype(np.str)
            for g in range(np.shape(out_dic)[0] - 1):
                out_dic[g, 0] = '%s-%s' % (out_dic[g, 0], out_dic[g + 1, 0])
            out_dic[g + 1, 0] = '%s-more' % out_dic[g + 1, 0]
            self.ExportData(out_dic, r'%s\dic_of_%d%s.txt' % (self.save_root, i, val_index[i, 1]))
        return use_data_raw

    def gener_dic(self,iv_group, woe_group, dic, iv_raw, i):
        '''
        the result of grouping discrete variable 离散型数据的分组结果
        the first column is group index   第一列为分组
        the second column is what values are in this group  第二列为对应的该分组的取值
        the third colmun is IV after grouping   第三列为该变量分组后的iv值，并保证分组后的iv/分组前的iv>0.8
        the fourth column is WOE of every group  第四列为该分组的woe
        '''
        if iv_raw[i] / iv_group[0] > 0.8:
            out = list(set(dic[:, 2]))
            out.sort()
        else:
            return []

        value_all = []
        for l in range(np.shape(out)[0]):
            value = []
            for m in range(np.shape(dic)[0]):
                if dic[m, 2] == out[l]:
                    value.append(dic[m, 0])
            value_all.append(value)

        iv_colm = []
        for l in range(np.shape(out)[0]):
            iv_colm.append(iv_group[0])

        woe_colm = []
        for l in range(np.shape(out)[0]):
            if out[l] == list(woe_group[0].keys())[l]:
                woe_colm.append(list(woe_group[0].values())[l])

        for l in range(np.shape(out)[0]):
            out[l] = int(out[l]) - 1

        out_dit = np.hstack(
            ((np.array(out)).reshape(np.shape(out)[0], 1), (np.array(value_all)).reshape(np.shape(value_all)[0], 1), \
             (np.array(iv_colm)).reshape(np.shape(iv_colm)[0], 1),
             (np.array(woe_colm)).reshape(np.shape(woe_colm)[0], 1)))
        return out_dit

    def flt_splt_single(self,i, use_data_raw,use_data_target):
        '''
        This is used to group one continuous variable     对连续性的变量进行分箱
        
        i is the volumn index of variable  为该变量在数据集中的列数
        use_data_raw is raw data set 初始的数据集
        '''

        group_i = use_data_raw[:, i].copy()

        # 第一步，将数据分为50组，阈值为分位数。
        # the 1st step, divided data into 50 groups and the thresholds are their percentile. May be group number is less than 50 , 
        # because reduplicated percentile.
        
        threshold = []
        for t in range(50):
            threshold.append(np.percentile(group_i.astype(np.float), t * 2))
        threshold = list(set(threshold))
        threshold.sort()

        group_i = self.replace_threshold(group_i, threshold)
        Woe = WOE(None)
        woe_group_1, iv_group_1 = Woe.woe(group_i.reshape(np.shape(group_i)[0], 1),use_data_target, event='1')
        iv_group_new = iv_group_1[0]  # iv the first step
        woe_i_sort = np.hstack(
            ((np.array(list(woe_group_1[0].keys()))).reshape(np.shape(np.array(list(woe_group_1[0].keys())))[0], 1), \
             (np.array(list(woe_group_1[0].values()))).reshape(np.shape(np.array(list(woe_group_1[0].keys())))[0], 1)))
        # just for sort
        woe_i_sort = woe_i_sort.astype(np.float).copy()
        woe_i_sort = woe_i_sort[np.lexsort(woe_i_sort[:, ::-1].T)]

        # 第二步，合并woe差距小的分组
        # the 2nd step, combine groups whose WOE are close
        
        gap = 0
        while (iv_group_new / iv_group_1)[0] >= 0.8:  
            # check whether it can be combined with the group that is prior 往小的数合并，并改为小的数
            for j in range(np.shape(woe_i_sort)[0] - 1):
                if abs(woe_i_sort[j + 1, 1] - woe_i_sort[j, 1]) <= gap:
                    woe_i_sort[j + 1, 0] = woe_i_sort[j, 0]
            threshold = list(set(woe_i_sort[:, 0]))
            threshold.sort()
            group_i = self.replace_threshold(group_i, threshold)
            # calculate the IV and WOE of group_iv 计算新group_i对应的iv和woe
            Woe = WOE(None)
            woe_group_new, iv_group_new = Woe.woe(group_i.reshape(np.shape(group_i)[0], 1),use_data_target, event='1')
            woe_i_sort = np.hstack(
                ((np.array(list(woe_group_new[0].keys()))).reshape(np.shape(np.array(list(woe_group_new[0].keys())))[0],
                                                                   1), \
                 (np.array(list(woe_group_new[0].values()))).reshape(
                     np.shape(np.array(list(woe_group_new[0].keys())))[0], 1)))
            woe_i_sort = woe_i_sort.astype(np.float).copy()
            woe_i_sort = woe_i_sort[np.lexsort(woe_i_sort[:, ::-1].T)]
            gap += 0.1
            if len(threshold) < 8 and len(threshold) >= 2:
                break
        return group_i, woe_i_sort

    def replace_threshold(self,group_i, threshold):
        # if a in [b,c) then change a to b   将数值改为他属于的分段的下限，左闭右开
        for g in range(np.shape(group_i)[0]):
            for t in range(np.shape(threshold)[0] - 1):
                if float(group_i[g]) >= threshold[t] and float(group_i[g]) < threshold[t + 1]:
                    group_i[g] = threshold[t]
                elif float(group_i[g]) >= threshold[-1]:
                    group_i[g] = threshold[-1]
        return group_i


    def ExportData(self,data, fileName):  # export
        f = open(fileName, 'w')
        for i in data:
            k = '\t'.join([str(j) for j in i])
            f.write(k + "\n")
        f.close()

