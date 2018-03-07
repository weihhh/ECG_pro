#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#读取TXT波形文件，检查数据长度及导联数

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt      #python画图包
import os

# test_list=[[1,5,7,12,3,12,3,56],[1,6,7,34,3,12,9,56]]
# test_array=np.array([[1,5,7,12,3,12,3,56],[1,6,7,34,3,12,9,56]])
# test_array_resape=test_array.reshape(-1)

# le =LabelEncoder()
# le.fit(test_array_resape)
# print('all classes: {}'.format(list(le.classes_)))

# for i in test_array:
#     step2=le.transform([[34,9,6],[9,9,6]])
# step2=le.transform([[34,9,6],[9,9,6]])
# print('test_transform: {}'.format(step2))
# # test_array_resape[0]=778
# # test_array_resape=test_array_resape.reshape([2,-1])
# # test_array=np.array([1,5,7,12,3,12,3,56])
# # test_array=np.array([1,5,7,12,3,12,3,56]).reshape([8,1])
# print(test_array_resape,'\n',test_array)


ECG_LEADS=['MDC_ECG_LEAD_V1','MDC_ECG_LEAD_V2','MDC_ECG_LEAD_V3','MDC_ECG_LEAD_V4','MDC_ECG_LEAD_V5','MDC_ECG_LEAD_V6','MDC_ECG_LEAD_I','MDC_ECG_LEAD_II','MDC_ECG_LEAD_III','MDC_ECG_LEAD_AVF','MDC_ECG_LEAD_AVL','MDC_ECG_LEAD_AVR']
ECG_CLASS=[1,2,3,5,12]
ECG_DATA_PATH=r'D:\aa_work\ECG\ECG_DATA_ALL\MIT_ECG_DATA'
ECG_LENGTH=15000

def transform_labels(labels):
    '''
    '''
    if not isinstance(labels,np.ndarray):
        print('Tranfer to array!')
        labels=np.array(labels)
    array_resape=labels.reshape(-1)
    le =LabelEncoder()
    le.fit(test_array_resape)
    print('all classes: {}'.format( list(le.classes_)))

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
    # if j==1:
    #     #normal
    #     i[no]=0
    # if j==2:
    #     #LBBB
    #     i[no]=1
    # if j==3:
    #     #RBBB
    #     i[no]=2
    # if j==5:
    #     #PVC
    #     i[no]=3
    # if j==12:
    #     #PACE
    #     i[no]=4


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
            return [int(i) for i in wave_list]#转换为int
        else:
            print('导联名称错误,不在导联列表中！---get_txt_wave()')
def get_txt_ann(filename):
    '''
    读取ann.txt文件
    time_list:标注的时间点int
    ann_list:标注int
    '''
    time_list=[]
    ann_list=[]

    with open(filename,'rt') as f:
        ann_list_str=f.readlines()
        for str_i in ann_list_str:
            ann_list.append(str_i.strip('\n').split(',')[1])
            time_list.append(str_i.strip('\n').split(',')[0])
    return [int(i) for i in time_list],[int(i) for i in ann_list]


def check_data(path,lead_list,leads_used='ALL'):
    '''
    '''
    if isinstance(leads_used,str) and leads_used.upper()=='ALL':
        leads_used=lead_list
    print('使用导联 ：',leads_used)
    wave_files=[filename for filename in get_files(path) if os.path.split(filename)[1].find('_wave')!=-1]
    ann_files=[filename.replace('_wave','_ann') for filename in wave_files ]

    err_list=[]
    for filename in wave_files:
        print('现在检查： {}'.format(filename))

        for lead_name in leads_used:
            wave_list=get_txt_wave(filename,lead_name,lead_list)
            if len(wave_list)!=ECG_LENGTH:
                err_list.append(filename)
    ann_counter=Counter()
    for filename in ann_files:
        #0-time,1-ann
        ann_counter.update(get_txt_ann(filename)[1])
    print('类别概览： ',sorted(ann_counter.items(),key=lambda x:x[0]))
    if not err_list:
        print('all right!')
    else:
        print('wrong length: {}'.format(err_list))

def network_input(path,lead_list,leads_used='ALL'):
    '''
    path:存储心电文件目录（全目录）
    lead_used:当前使用的导联,输入'all'或'ALL'表示使用全部
    lead_list：导联默认顺序
    return：列表，其中为各记录中的导联波形数据（字典形式存储）

    '''
    if isinstance(leads_used,str) and leads_used.upper()=='ALL':
        leads_used=lead_list
    print('使用导联 ：',leads_used)
    wave_files=[filename for filename in get_files(path) if os.path.split(filename)[1].find('_wave')!=-1]

    # wave_data_list=[]
    for filename in wave_files:
        #字典存储使用的导联，一个lead_dict中包括了所有使用的导联数据和对应的标注和标注时间
        print('现在处理： {}'.format(filename))
        lead_dict={}
        for lead_name in leads_used:
            wave_list=get_txt_wave(filename,lead_name,lead_list)
            lead_dict[lead_name]=wave_list
            ann=get_txt_ann(filename.replace('_wave','_ann'))
            lead_dict['ann_time']=ann[0]
            #进行了标签转换
            lead_dict['ann_str']=transform_list_lables(ann[1],ECG_CLASS)
        # wave_data_list.append(lead_dict)
        yield lead_dict

def main():

    #生成器，验证导联数目，
    # all_data=network_input(ECG_DATA_PATH,ECG_LEADS)
    # for file_data in all_data:
    #     # print('数据字典属性： {}'.format(file_data.keys()))
    #     assert(len(file_data.keys())==14)
    #     for lead_name,i in [i for i in file_data.items() if i[0] not in ['ann_time','ann_str']]:
    #         assert(len(i)==ECG_LENGTH)
            # plt.plot(i)
            # plt.show()
    time_list,ann_list=get_txt_ann(r'D:\aa_work\ECG\ECG_DATA_ALL\ECG_DATA'+r'\201801051344_03759378_金祖行_ann.txt')
    print(ann_list)
    print(transform_list_lables(ann_list,[1,2,3,5,12]))
    # transform_labels(test_list)
    check_data(ECG_DATA_PATH,ECG_LEADS)
if __name__ == '__main__':
  main()