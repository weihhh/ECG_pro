#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#10个心拍为一个输入样本，长度进行归一化处理（2880）


import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import io
from scipy import stats
import pickle
from collections import Counter

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

def adjust_len(input,target_len):
    #input array
    while len(input)<target_len:
        now_len=len(input)
        x=np.random.randint(1,now_len-1)
        insert_value=(input[x-1]+input[x])/2
        input=np.insert(input,x,insert_value)
    while len(input)>target_len:
        now_len=len(input)
        x=np.random.randint(1,now_len-1)
        input=np.delete(input,x)
    return input
#注释mat文件列表
files_list=[i for i in get_files(path) if os.path.basename(i).rfind('ANNOTD.mat')!=-1]
# print(files_list[0],os.path.basename(files_list[0])[:3])


all_ecg_list=[]
all_ann_list=[]
all_file_list=[]
all_len_list=[]
for file in files_list:
    index=os.path.basename(file)[:3]
    print('处理文件序号： ',index) 
    
    all_file_list.append(index)

    #波形数据矩阵
    data_m=load_mat(path,'{}.mat'.format(index))
    ECG_matrxi=data_m['M']#array(648000,2),因matlab中存放的矩阵名叫M
    if index=='114':
        ii_sample=ECG_matrxi[:,1]#(648000,)
    else:
        ii_sample=ECG_matrxi[:,0]#(648000,)
    #print type(ii_sample)
    
    #print 'ii_sample: ',ii_sample.shape

    #注释
    annotation=load_mat(path,'{}ANNOTD.mat'.format(index))
    annotation_matrix=annotation['ANNOTD']#(2266，1)
    
    annotation_counts=Counter(annotation_matrix[:,0])
    #print '标注类别及个数： ',annotation_counts 


    #注释矩阵对应时间
    annotation_time=load_mat(path,'{}ATRTIMED.mat'.format(index))
    annotation_time_matrix=annotation_time['ATRTIMED']#(2266，1)
    #将时间转换为对应采样点个数
    annotation_sampleno=[(R_time*360).astype(int) for R_time in annotation_time_matrix.flatten()]

    #保存单个文件波形
    file_ecg_list=[]
    file_ann_list=[]
    #每11个标签截取10段
    seg=10
    for i in range(0,len(annotation_matrix),seg):
        #截取的段
        starti=i
        endi=i+seg
        if endi>=len(annotation_matrix):
            break
        #显式copy
        ecg_wait=ii_sample[annotation_sampleno[starti]:annotation_sampleno[endi]].copy()
        ecg_final=adjust_len(ecg_wait,2880)
        file_ecg_list.append(ecg_final)
        file_ann_list.append(list(annotation_matrix.flatten()[starti:endi]))
    all_ecg_list.append(file_ecg_list)
    all_ann_list.append(file_ann_list)


# plt.plot(all_ecg_list[1][3])
# plt.show()

for file in all_ann_list:
    for i in file:
        for no,j in enumerate(i):
            # print(type(j),j)
            if j not in [1,2,3,5,12]:
                i[no]=5
            else:
                if j==1:
                    #normal
                    i[no]=0
                if j==2:
                    #LBBB
                    i[no]=1
                if j==3:
                    #RBBB
                    i[no]=2
                if j==5:
                    #PVC
                    i[no]=3
                if j==12:
                    #PACE
                    i[no]=4



#pickle
with open('ECG_10segment_data.pickle','wb') as f:
    pickle.dump([all_ecg_list,all_ann_list,all_file_list],f)

