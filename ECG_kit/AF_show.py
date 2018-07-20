#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    Show info of dataset
'''
import scipy.io
import numpy as np
from collections import Counter
import platform
#方便部署服务器
if 'Linux'==platform.system():
    #终端运行需要修改绘图后端
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

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
    

def show_trainset(trainset,traintarget):
    print('trainset: ',trainset.shape,'traintarget: ',traintarget.shape)
    ytrue=np.argmax(traintarget,axis=1)

    ann_counter=Counter(ytrue)
    counter_result=sorted(ann_counter.items(),key=lambda x:x[0])
    print('类别概览： ',counter_result)
    return counter_result

def wgn(x, snr):
    # x=np.array(x)
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)



# wave=trainset[0][:2000]
# plt.plot(wave)
# wave_noise=wgn(wave,30)
# wgn_wave=wave_noise+wave
# plt.figure()
# plt.plot(wgn_wave)
# plt.show()




