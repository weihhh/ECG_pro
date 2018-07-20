#!/usr/bin/env python3
# -*- coding: utf-8 -*-
_author_='weizhang'
version='V1.1'

#########################
#V1
#1.统一从类别目录中取出数据
#2.全部放入内存，速度更快

#v1.1
# 1.从目录中取出数据文件后进行排序，而不是打乱,get_files函数中
#########################
import scipy.io
import numpy as np
import glob,platform,os,sys
from sklearn.utils import shuffle

import ECG_Uni_Data

FS=300
Time_s=60
MAX_LEN = Time_s*FS
CLASSES = ['A', 'N', 'O','~']
ECG_LEADS=['MDC_ECG_LEAD_I','MDC_ECG_LEAD_II','MDC_ECG_LEAD_III','MDC_ECG_LEAD_AVR','MDC_ECG_LEAD_AVL','MDC_ECG_LEAD_AVF','MDC_ECG_LEAD_V1','MDC_ECG_LEAD_V2','MDC_ECG_LEAD_V3','MDC_ECG_LEAD_V4','MDC_ECG_LEAD_V5','MDC_ECG_LEAD_V6']
Lead='MDC_ECG_LEAD_I'
# CurrentPath=os.getcwd()
CurrentPath='/home/ubuntu417/ECG/test63/baseline_5fold'

def input_matrix(filenames,labels):
    '''
        返回矩阵samples*MAX_LEN,array labels,其中为'A','N'这样的原始标签
    '''
    # 用于放置数据的数组
    trainset = np.zeros((len(filenames),MAX_LEN))
    for count,file in enumerate(filenames):
        wave=ECG_Uni_Data.get_txt_wave(file,Lead,ECG_LEADS)
        data = np.nan_to_num(wave)

        #标准化
        data = data - np.mean(data)
        data = data/np.std(data)
        trainset[count,:min(MAX_LEN,len(data))] = data[:min(MAX_LEN,len(data))]
    return trainset,np.array(labels)

def get_files():
    all_files=[]
    labels=[]
    for class_now in CLASSES:
        if os.path.exists(os.path.join(CurrentPath,class_now)):
            files = sorted(glob.glob(os.path.join(CurrentPath,class_now,r'*wave*.txt')))
            all_files+=files
            labels+=[class_now]*len(files)
    # all_files,labels=shuffle(all_files,labels)
    filenames=[os.path.split(i)[1] for i in all_files]
    all_files,labels=np.array(all_files),np.array(labels)
    sorted_arg=np.argsort(filenames)
    all_files,labels=all_files[sorted_arg],labels[sorted_arg]
    print(all_files[:10],labels[:10])
    return all_files,labels

def input_for_resnet():
    all_files,labels=get_files()
    trainset,labels=input_matrix(all_files,labels)
    traintarget = np.zeros((trainset.shape[0],4))
    for i,label in enumerate(labels):
        #所属类别位置置一，相当于one_hot
        traintarget[i,CLASSES.index(label)] = 1
    # Saving both
    scipy.io.savemat('trainingset.mat',mdict={'trainset': trainset,'traintarget': traintarget})

if __name__ == '__main__':
    input_for_resnet()