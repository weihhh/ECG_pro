#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#将MIT_mat文件中的原始ECG数据，重写为项目约定格式

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import io
from scipy import stats

path=r'D:\aa_work\新学期项目及论文\心电图项目\数据集\MIT-BIH Arrhythmia Database\mat_data'

def get_files(path,recurse=False):
    '''
    得到指定目录下的所有文件列表，递归或者不递归
    '''
    files=[]
    if not recurse:
        for name in os.listdir(path) :
            fullname=os.path.join(path,name)
            if os.path.isfile(fullname):
                files.append(fullname)
        return files   
    else:
        for root ,dirs,names in os.walk(path) :
            for name in names:
                fullname=os.path.join(root,name)
                files.append(fullname)
        return files

def load_mat(folder,file_name):
    #loaddata
    folder=os.path.normcase(folder+'/')
    # folder='/Users/jig289/Dropbox/MATLAB/Projects/In_Progress/BMI/Processed_Data/' 
    data=io.loadmat(folder+file_name)
    return data

files_list=[i for i in get_files(path) if os.path.basename(i).rfind('ANNOTD.mat')!=-1]
#这里控制处理几个文件
for file in files_list[:3]:
    index=os.path.basename(file)[:3]
    print('处理文件序号： ',index) 
    
    #波形数据矩阵
    data_m=load_mat(path,'{}.mat'.format(index))
    ECG_matrxi=data_m['M']#array(648000,2),因matlab中存放的矩阵名叫M
    if index=='114':
        ii_sample=ECG_matrxi[:,1]#(648000,)
    else:
        ii_sample=ECG_matrxi[:,0]#(648000,)

    #注释
    annotation=load_mat(path,'{}ANNOTD.mat'.format(index))
    annotation_matrix=annotation['ANNOTD'].flatten()#(2266，1)
    
    #注释矩阵对应时间
    annotation_time=load_mat(path,'{}ATRTIMED.mat'.format(index))
    annotation_time_matrix=annotation_time['ATRTIMED']#(2266，1)
    annotation_sampleno=[(R_time*360).astype(int) for R_time in annotation_time_matrix.flatten()]
    
    if not os.path.exists(os.path.join(os.getcwd(),'MIT_ECG_DATA')):
        os.mkdir('./MIT_ECG_DATA')
    print('MIT_ECG_DATA in :{}'.format(os.getcwd()))
    #写入波形数据
    with open(os.path.join(os.getcwd(),'MIT_ECG_DATA',index+'mit_wave.txt'),'wt') as f:
        for i,data in enumerate(ii_sample):
            if i ==0:
                f.write(str(data))
            else:
                f.write(',')
                f.write(str(data))
    #写入标注文件
    with open(os.path.join(os.getcwd(),'MIT_ECG_DATA',index+'mit_ann.txt'),'wt') as f:
        for time,ann in zip(annotation_sampleno,annotation_matrix):
            f.write(str(time)+',')
            f.write(str(ann)+'\n')
