#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#10s为一个输入样本，每一秒一个标注，
#根据不正常心拍与这一秒中心的距离决定当前心拍类型，共5类


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


#注释mat文件列表
files_list=[i for i in get_files(path) if os.path.basename(i).rfind('ANNOTD.mat')!=-1]
# print(files_list[0],os.path.basename(files_list[0])[:3])


all_ecg_list=[]
all_ann_list=[]
all_file_list=[]
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
    annotation_sampleno=[(R_time*360).astype(int) for R_time in annotation_time_matrix.flatten()]

    #每10s截取一段数据，添加标签
    end_of_segment=3600
    #保存单个文件波形
    file_ecg_list=[]
    file_ann_list=[]
    for start_i in range(0,len(ii_sample),3600):
        end_i=start_i+end_of_segment
        file_ecg_list.append(ii_sample[start_i:end_i])
        tag_list=[]
        end_of_segmentj=360
        for j in range(start_i,end_i,360):
            tag=None
            min_dis=360
            end_j=j+end_of_segmentj
            for i,item in enumerate(annotation_sampleno):
                if item >end_j:
                    break
                if item<=end_j and item>=j and annotation_matrix.flatten()[i] !=1:
                    if np.fabs(item-(j+180))<min_dis:
                        min_dis=np.fabs(item-(j+180))
                        tag=annotation_matrix.flatten()[i]
            if tag==None:
                tag=1
            tag_list.append(tag)
        file_ann_list.append(tag_list)
    #all_ecg_list， all_ann_list，长度46，即文件个数,file_ecg_list为array的list
    all_ecg_list.append(file_ecg_list)
    all_ann_list.append(file_ann_list)


for file in all_ann_list:
    for i in file:
        for no,j in enumerate(i):
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

print('shape of all_ecg_list: {}'.format(len(all_ecg_list)))
print('shape of all_ann_list: {}'.format(len(all_ann_list)))

print(np.array(all_ann_list).shape)
print(np.array(all_ann_list).flatten().shape)
all_counts=Counter(np.array(all_ann_list).flatten())
print(all_counts)

#pickle
# with open('ECG_10S_data.pickle','wb') as f:
#     pickle.dump([all_ecg_list,all_ann_list,all_file_list],f)

#拼接
# all_spike_matrix=np.concatenate(all_ecg_list,axis=0)
# all_ann_matrix=np.concatenate(all_ann_list,axis=0)
# print(all_spike_matrix.shape,all_ann_matrix.shape)

# #give matlab
# # io.savemat('ECGdata.mat',{'ECGdata':new_data,'index':new_index})


