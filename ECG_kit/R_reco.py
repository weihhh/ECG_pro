#R峰识别程序

import pywt
import math,os
import numpy as np
import matplotlib.pyplot as plt

def R_reco_wlt(wave_list,ini_threshold=1.0723,R_R=234.300):
    '''
    wavelet R peak recognition
    '''
    A=pywt.cwt(wave_list,3, 'mexh')
    coefs = A[0].tolist()[0]
    '''
    plt.figure()
    plt.plot(cofes[0:1000])
    plt.show()
    '''

    freq=360;
    num=len(wave_list);
    max_sum=0;
    max_mean=0;
    maximal=[];  #极大值
    maxaddr=[];  #极大值时间点
    space=0;     #不应期200ms，初始化0


    ptr_max=0;
    count = 0;  #为了计算一个R峰后，未检测出R峰的时间，补偿法的变量
    halfA=ini_threshold/2;
    k=0;
    for i in range(num):
        if space != 0:
            space = space - 1;
        if coefs[i]>ini_threshold and space == 0 :
          if coefs[i]>coefs[i-1] and coefs[i]>coefs[i+1]:
            space=72;    #不应期200ms
            ptr_max=coefs[i];
            maximal.append(coefs[i]);
            maxaddr.append(i);
            ini_threshold = 0.75*ini_threshold + 0.25*(5/9)*coefs[i];   # 实现自适应阈值法
            count=0;

        count=count+1;
        ### 实现补偿法策略
        if(count>=1.5*R_R):
            halfA=ini_threshold/2;
            re=math.floor(i-1.1*R_R);
            for j in range(math.floor(1.1*R_R)):
                if coefs[re+j]>halfA and space == 0:
                    if coefs[re+j]>coefs[re+j-1] and coefs[re+j]>=coefs[re+j+1]:
                        space = 72;
                        maximal.append(coefs[re+j]);
                        maxaddr.append(re+j);
                        ini_threshold = 0.75*ini_threshold + 0.25*(5/9)*coefs[re+j];
                        k=k+1;      ##统计通过补偿法得到的R峰值点
                        count=0;
                        break;
    return maxaddr

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


# Current_path=os.getcwd()
# if not os.path.exists(os.path.join(Current_path,'MIT_ECG_DATA')):
#     print('no MIT_ECG_DATA!')
#     exit()
# files_list=[file for file in get_files(os.path.join(Current_path,'MIT_ECG_DATA')) if os.path.split(file)[1].find('_data')!=-1]
# for file in files_list:
#     print(file)
#     with open(file,'rt') as f:
#         #注意这里读取的时候要去掉首尾换行符
#         data_list=f.readlines()[0].strip('\n').split(',')#读取整个文件所有行，保存在一个列表(list)变量中，每行作为一个元素
#     ann_time=R_reco_wlt(data_list)
#     #写入标注文件
#     with open(os.path.splitext(file)[0]+'_test.txt','wt') as f:
#         for time in ann_time:
#             f.write(str(time)+',')
#             f.write('1'+'\n')

def R_R_THRESHOLD():
    pass
