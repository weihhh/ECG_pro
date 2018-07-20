#!/usr/bin/env python3
# -*- coding: utf-8 -*-
_author_='weizhang'
version='V0'

###################################
#v0:
#1.提供读取txt格式数据,转换为mat格式
###################################

import glob,platform
import os,sys,shutil
from collections import Counter
import numpy as np
import csv
import scipy.io as sio
#方便部署服务器
if 'Linux'==platform.system():
    #终端运行需要修改绘图后端
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

ECG_LEADS=['MDC_ECG_LEAD_I','MDC_ECG_LEAD_II','MDC_ECG_LEAD_III','MDC_ECG_LEAD_AVR','MDC_ECG_LEAD_AVL','MDC_ECG_LEAD_AVF','MDC_ECG_LEAD_V1','MDC_ECG_LEAD_V2','MDC_ECG_LEAD_V3','MDC_ECG_LEAD_V4','MDC_ECG_LEAD_V5','MDC_ECG_LEAD_V6']
HOS_LEADS=['MDC_ECG_LEAD_I','MDC_ECG_LEAD_II']

def sub_sample(input_signal,src_fs,tar_fs):
    '''

    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    '''
    dtype = input_signal.dtype
    audio_len = len(input_signal)
    audio_time_max = 1.0*(audio_len-1) / src_fs
    src_time = 1.0 * np.linspace(0,audio_len,audio_len) / src_fs
    tar_time = 1.0 * np.linspace(0,np.int(audio_time_max*tar_fs),np.int(audio_time_max*tar_fs)) / tar_fs
    output_signal = np.interp(tar_time,src_time,input_signal).astype(dtype)
    return output_signal

def get_txt_wave(filename,lead_name,lead_list):
    '''
    filename：记录名称
    lead_name:想获得的导联
    lead_list:导联列表，展示顺序
    return： wave_data_list，已转换为int类型
    '''
    with open(filename,'rt') as f:
        #注意这里读取的时候要去掉首尾换行符
        lead_index=lead_list.index(lead_name)
        if lead_index!=-1:
            wave_list=f.readlines()[lead_index].strip('\n').split(',')#读取整个文件所有行，保存在一个列表(list)变量中，每行作为一个元素
            wave_list=[int(i) for i in wave_list]#转换为int
            if not wave_list:
                print('{} 导联为空！<---{}'.format(lead_name,sys._getframe().f_code.co_name))
            return wave_list
        else:
            print('导联名称错误,不在导联列表中！<---{}'.format(sys._getframe().f_code.co_name))

def TXT2WFDB(source_file,target_path,src_fs,tar_fs,args=None):
    '''
        source_file:绝对路径
        target_path:目标存储路径
        默认500hz，I导联
    '''
    LEAD_READ=0#读取第二行，I导联
    record_name=os.path.basename(os.path.splitext(source_file)[0])
    record_file=record_name+'.mat'
    print('record_name: ',record_name)
    if not args:
        lead_num='1'#默认为1，暂时不考虑多导联存储
        sample_rate='300'
        data=np.array(get_txt_wave(source_file,'MDC_ECG_LEAD_I',ECG_LEADS),dtype=np.int16)
        if src_fs!=tar_fs:
            data=sub_sample(data,src_fs,tar_fs)
        sample_length=str(len(data))
        date='2013-12-23'
        time='08:12:08 '
        data_format='16+24'
        calibration='1000/mV'
        x1='16'
        x2='0'
        x3='-188'
        x4='0'
        x5='0'
        x6='ECG '#注意这里一个空格

    else:
        #将args填充到参数中
        pass
    with open(os.path.join(target_path,record_name+'.hea'),'w') as f:
        f.write(' '.join([record_name,lead_num,sample_rate,sample_length,date,time]))
        f.write('\n')
        f.write(' '.join([record_file,data_format,calibration,x1,x2,x3,x4,x5,x6]))
        f.write('\n')
    #mat 数据
    # file = open(source_file)
    # lines = file.readlines()
    # data = lines[LEAD_READ].strip().split(',')
    # data =np.array(list(map(int, data)),dtype=np.int16)
    sio.savemat(os.path.join(target_path,record_file), {'val':data},format = '4')
    print('finish!')

def DIR_TXT2WFDB(source_dir,target_dir):
    if 'Linux'==platform.system():
        files = sorted(glob.glob(source_dir+r"/*.txt"))
    else:
        files = sorted(glob.glob(source_dir+r"\*.txt"))
    for file in files:
        TXT2WFDB(file,target_dir,500,300)

if __name__ == '__main__':
    DIR_TXT2WFDB('D:\ECG\ECG_DATA_ALL\HOS_ECG_DATA','D:\ECG\ECG_DATA_ALL\HOS_MAT_DATA')