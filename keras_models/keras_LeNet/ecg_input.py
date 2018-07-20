#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#统一txt格式数据输入
import numpy as np

import Get_txt_data
from collections import Counter
from sklearn.model_selection import train_test_split#训练数据、测试数据切分

ECG_LEADS=['MDC_ECG_LEAD_V1','MDC_ECG_LEAD_V2','MDC_ECG_LEAD_V3','MDC_ECG_LEAD_V4','MDC_ECG_LEAD_V5','MDC_ECG_LEAD_V6','MDC_ECG_LEAD_I','MDC_ECG_LEAD_II','MDC_ECG_LEAD_III','MDC_ECG_LEAD_AVF','MDC_ECG_LEAD_AVL','MDC_ECG_LEAD_AVR']
ECG_CLASS=[1,2,3,5,12]
ECG_DATA_PATH=r'D:\aa_work\ECG\ECG_DATA_ALL\ECG_DATA'
ECG_LENGTH=15000
SAMPLE_RATE=500

##单心拍(300)样本生成,II导联##
def gen_single_beat():
    '''

    '''
    file_dict_list=Get_txt_data.network_input(ECG_DATA_PATH,ECG_LEADS,['MDC_ECG_LEAD_II'])
    wave_input=[]
    ann_input=[]
    segment_log=[]#记录每个文件内心拍个数序号
    segment_log_i=0
    for file_dict in file_dict_list:

        end_no=len(file_dict['MDC_ECG_LEAD_II'])-1#尾sample序号
        right=len(file_dict['ann_time'])
        left=0
        for i,R_samples in enumerate(file_dict['ann_time']):
            begin=R_samples-149
            # print(begin)
            end=R_samples+151
            # print(end)
            if begin>=0 and end<=end_no:
                wave_input.append(file_dict['MDC_ECG_LEAD_II'][begin:end])#300个点
                ann_input.append(file_dict['ann_str'][i])
                segment_log_i+=1
                continue
            if begin<0:
                left=i+1
                continue
            if end>end_no:
                right=i
                continue
        segment_log.append(segment_log_i)

    wave_input=np.array(wave_input)
    ann_input=np.array(ann_input)
    return wave_input,ann_input

def check_ndarray(x):
    print('shape: {}'.format(x.shape))
    print('size: {}'.format(x.size))
    print('type: {}'.format(x.dtype))
    print('itemsize(byte): {}'.format(x.itemsize))

def get_data():
    wave_input,ann_input=gen_single_beat()
    print('原始数据规模： ',wave_input.shape,'原始标签规模： ',ann_input.shape)
    annotation_counts=Counter(ann_input)
    print('类别概览： ',annotation_counts)

    #未归一化!

    X_train,X_test, y_train,y_test=train_test_split(wave_input,ann_input,test_size=0.2)
    print('训练集数据：')
    check_ndarray(X_train)
    print('训练集标签：')
    check_ndarray(y_train)
    print('测试集数据：')
    check_ndarray(X_test)
    print('测试集标签：')
    check_ndarray(y_test)
    return (X_train, y_train), (X_test, y_test)



def main():
    
    get_data()
if __name__ == '__main__':
  main()

