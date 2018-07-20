#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    augument dataset
'''

import scipy.io
import numpy as np
import AF_show

ORI_trainset='trainingset.mat'
AUG_trainset='trainingset_augu.mat'

### gussian noise
def wgn(x, snr):
    # x=np.array(x)
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def augument_ecg(ecg_list,ann_list,length,aug_lag,aug_times):
    '''
        move aug_lag segment,gussian noise,300hz,aug_times times aug
        return length=length-aug_lag
    '''
    augu_ecg_list=[]
    augu_ann_list=[]
    for wave_i,wave in enumerate(ecg_list):
        if np.argmax(ann_list[wave_i])==0 :
            for i in range(0,aug_lag,aug_lag//aug_times):
                wave_select=wave[i:i+length-aug_lag].copy()
                if i!=0:
                    wave_noise=wgn(wave_select,30) 
                    wave_select=wave_select+wave_noise
                augu_ecg_list.append(wave_select)
                augu_ann_list.append(ann_list[wave_i].copy())
        elif np.argmax(ann_list[wave_i])==3:
            for i in range(0,aug_lag,aug_lag//(aug_times*2)):
                wave_select=wave[i:i+length-aug_lag].copy()
                if i!=0:
                    wave_noise=wgn(wave_select,30) 
                    wave_select=wave_select+wave_noise
                augu_ecg_list.append(wave_select)
                augu_ann_list.append(ann_list[wave_i].copy())
        else:
            augu_ecg_list.append(wave[0:length-aug_lag].copy())
            augu_ann_list.append(ann_list[wave_i].copy())
    return augu_ecg_list,augu_ann_list

if __name__ == '__main__':

    print('Loading dataset-- {} !'.format(ORI_trainset))
    matfile=scipy.io.loadmat(ORI_trainset)
    trainset=matfile['trainset']
    traintarget=matfile['traintarget']

    print('Before augu ！')
    counter_result=AF_show.show_trainset(trainset,traintarget)
    #show_bar_from_counter(counter_result)

    ####################### 开始增强

    augu_ecg_list,augu_ann_list=augument_ecg(trainset,traintarget,18000,305)
    augu_ecg_list=np.array(augu_ecg_list)
    augu_ann_list=np.array(augu_ann_list)
    scipy.io.savemat('trainingset_augu.mat',mdict={'trainset': augu_ecg_list,'traintarget': augu_ann_list})

    # ###################### 增强后
    print('Loading augu dataset-- {} !'.format(AUG_trainset))
    matfile=scipy.io.loadmat(AUG_trainset)
    trainset=matfile['trainset']
    traintarget=matfile['traintarget']

    print('After augu ！')
    counter_result=AF_show.show_trainset(trainset,traintarget)
    AF_show.show_bar_from_counter(counter_result)

    # wave=trainset[0][:2000]
    # plt.plot(wave)
    # wave_noise=wgn(wave,30)
    # wgn_wave=wave_noise+wave
    # plt.figure()
    # plt.plot(wgn_wave)
    # plt.show()