#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#统一txt格式数据输入
import numpy as np

import Get_txt_data
from collections import Counter

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

def data_preprocess(wave_input,ann_input):
    print('原始数据规模： ',wave_input.shape,'原始标签规模： ',ann_input.shape)
    annotation_counts=Counter(ann_input)
    print('类别概览： ',annotation_counts)
    pass

def main():
    wave_input,ann_input=gen_single_beat()
    data_preprocess(wave_input,ann_input)
if __name__ == '__main__':
  main()

