#!/usr/bin/env python3
# -*- coding: utf-8 -*-
_author_='weizhang'
version='V0'

###################################
#v0:
#1.提供读取txt格式数据的函数接口get_txt_wave
#2.转换标签函数
#3.将此文件放在数据目录下，汇总波形信息函数
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


if 'HOS' in os.getcwd():
    Default_FS=500
    TAR_FS=300
else:
    Default_FS=300
    TAR_FS=300
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


def TXT2WFDB(source_file,target_path,args=None):
    '''
        source_file:绝对路径
        target_path:目标存储路径
        默认500hz，I导联
    '''
    LEAD_READ=0#读取第二行，I导联
    Default_FS=300
    TAR_FS=300
    record_name=os.path.basename(os.path.splitext(source_file)[0])
    record_file=record_name+'.mat'
    print('record_name: ',record_name)
    if not args:
        lead_num='1'#默认为1，暂时不考虑多导联存储
        sample_rate='300'
        data=get_txt_wave(source_file,'MDC_ECG_LEAD_I',ECG_LEADS)
        data=sub_sample(data,Default_FS,TAR_FS)
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
    with open(record_name+'.hea','w') as f:
        f.write(' '.join([record_name,lead_num,sample_rate,sample_length,date,time]))
        f.write('\n')
        f.write(' '.join([record_file,data_format,calibration,x1,x2,x3,x4,x5,x6]))
        f.write('\n')
    #mat 数据
    file = open(source_file)
    lines = file.readlines()
    data = lines[LEAD_READ].strip().split(',')

    data =np.array(list(map(int, data)),dtype=np.int16)
    sio.savemat(os.path.join(target_path,record_file), {'val':data},format = '4')
    print('finish!')

def TXT2WFDB_DIR(source_dir,target_dir):
    '''
        source_dir:绝对路径
        target_dir:目标存储路径
        默认300hz，I导联
    '''
    wave=sub_sample(wave,Default_FS,TAR_FS)

def transform_list_lables(labels,labels_used):
    '''
    labels：待转换labels
    label_used:将label转换为label在label_used中的序号
    labels：list，其中标注为int
    '''
    switcher={}
    for no,label in enumerate(labels_used):
        switcher[label]=no
    print('类别： ',switcher,end='')
    print('其他：{}'.format(len(labels_used)))
    for no,j in enumerate(labels):
        j=int(j)
        #字典中没有的话就设置为len(labes_used)，表示其他
        labels[no]=switcher.get(j,len(labels_used))
    return labels

def show_data_summary():
    dataDir=os.getcwd()
    ## Loading time serie signals
    if 'Linux'==platform.system():
        files = sorted(glob.glob(dataDir+r"/*.txt"))
    else:
        files = sorted(glob.glob(dataDir+r"\*.txt"))
    #排除标注文件
    files=[file for file in files if 'ann' not in file]
    leads_sta=[{'length':[],'wave_amp':[],'files_num':0} for i in range(len(ECG_LEADS))]
    for file in files:
        for i,lead in enumerate(ECG_LEADS):
            wave=get_txt_wave(file,lead,ECG_LEADS)
            if len(wave)>1:
                leads_sta[i]['length'].append(len(wave))
                leads_sta[i]['wave_amp'].append(np.mean(wave))
                leads_sta[i]['files_num']+=1
    for lead,statics in zip(ECG_LEADS,leads_sta):
        if not statics['length']:
            continue
        #长度分布
        len_result={'min_than_{}'.format(15):0,'between{}-{}'.format(15,30):0,'between{}-{}'.format(30,45):0,'max_than_{}'.format(45):0}
        for i in statics['length']:
            if i <15*Default_FS:
                len_result['min_than_{}'.format(15)]+=1
            elif i<30*Default_FS:
                len_result['between{}-{}'.format(15,30)]+=1
            elif i<45*Default_FS:
                len_result['between{}-{}'.format(30,45)]+=1
            else:
                len_result['max_than_{}'.format(45)]+=1
        #幅值平均值
        amp_mean=np.mean(statics['wave_amp'])
        amp_median=np.median(statics['wave_amp'])
        #文件数目
        files_num=statics['files_num']
        print('lead: ',lead)
        print('length: ',len_result)
        print('amp_mean: ',amp_mean)
        print('amp_median: ',amp_median)
        print('files_num: ',files_num)

def show_classes_distribution(ytrue):
    '''
    '''
    ann_counter=Counter(ytrue)
    counter_result=sorted(ann_counter.items(),key=lambda x:x[0])
    print('类别概览： ',counter_result)
    return counter_result

def show_bar_from_counter(counter):
    data=[i[1] for i in counter]
    labels=['A', 'N', 'O', '~']
    rects=plt.bar(labels, data)
    for rect in rects:  
        height = rect.get_height()  
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')  
    
    plt.savefig('counter.png', format='png', dpi=1000)
    if 'Linux'!=platform.system():
        plt.show()

def save_result(ytrue,ypred,testname):
    #write to csv
    with open('submission_{}.csv'.format(testname), 'w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for i,j in zip(ytrue,ypred):
            spamwriter.writerow([i, j])
def read_ann(annfilename):
    ann_file=[]
    ytrue=[]
    with open(os.path.join(os.getcwd(),annfilename),'r') as f:
        for line in f.readlines():
            anns=line.strip('\n').split(',')
            ann_file.append(anns[0])
            ytrue.append(anns[1])
    return ann_file,ytrue

def split_class(target_path=os.getcwd()):
    '''
        target_path:选择哪个目录下放置类别目录
    '''
    if Default_FS==300:
        ann_file,ytrue=read_ann('AF_ann.txt')
    if Default_FS==500:
        ann_file,ytrue=read_ann('hos_ann.txt')
    counter_result=show_classes_distribution(ytrue)
    calsses=[i[0] for i in counter_result]
    for ecg_class in calsses:
        if not os.path.exists(os.path.join(target_path,ecg_class)):
            print('creat {} dir!'.format(ecg_class))
            os.mkdir(os.path.join(target_path,ecg_class))
    for file,label in zip(ann_file,ytrue):
        #根据HOS是否在路径中判断是医院数据还是AF数据
        if Default_FS==300:
            shutil.copyfile(file+'_wave.txt',os.path.join(target_path,label,file+'.txt'))
        else:
            #HOS 数据需要降采样！！！！！！
            exits_leads=[]
            for lead in HOS_LEADS:
                wave=np.array(get_txt_wave('hos_wave'+file+'.txt',lead,ECG_LEADS))
                wave=sub_sample(wave,Default_FS,TAR_FS)
                exits_leads.append((lead,wave))
            with open(os.path.join(target_path,label,'hos_wave'+file+'.txt'),'wt') as f:
                #暂时补全为12导联，II导联处于第二行
                #I导联
                for i,data in enumerate(exits_leads[0][0]):
                    if i ==0:
                        f.write(str(data))
                    else:
                        f.write(',')
                        f.write(str(data))
                f.write('\n')
                #II导联
                for i,data in enumerate(exits_leads[1][0]):
                    if i ==0:
                        f.write(str(data))
                    else:
                        f.write(',')
                        f.write(str(data))

                for i in range(10):
                    f.write('\n'+'0')

if __name__ == '__main__':
    show_data_summary()
    split_class(r'/home/ubuntu417/ECG/test531')