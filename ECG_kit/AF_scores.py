#!/usr/bin/env python3
# -*- coding: utf-8 -*-
_author_='weizhang'
Version='V3'

##########################
#统一所有AF程序的评价模块
#全部使用混淆矩阵形式 
#V2
#可以使用此模块处理针对训练集测试和交叉验证两种情况的混淆矩阵

#V3
#加入 save_result 函数，将预测结果存储,scores 函数增加testname默认参数
##########################

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

import itertools,time
import os
import csv

#flag  True表示进行模块验证,False 表示读取本目录下confusion.mat并显示结果
Test=False

classes = ['A', 'N', 'O', '~']


def save_result(ytrue,ypred,testname):
    #write to csv
    with open('submission_{}.csv'.format(testname), 'w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for i,j in zip(ytrue,ypred):
            spamwriter.writerow([i, j])

def scores(ytrue, ypred,testname='default'):
    '''
        ytrue:true lables
        ypred:model predict lables
    '''
    save_result(ytrue,ypred,testname)
    timenow=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    timenow_forfile=time.strftime("%Y_%m_%d#%H_%M_%S", time.localtime())

    if ytrue is not None:
        report=classification_report(ytrue,ypred,digits=4)
        print(report)
        cv=confusion_matrix(ytrue, ypred)
        # Saving cross validation results 
        print('save confusion_{}.mat!'.format(timenow_forfile))
        scipy.io.savemat('confusion_{}.mat'.format(timenow_forfile),mdict={'cvconfusion': cv.tolist()})
        print('confusion matrix: \n {}'.format(cv))
    else:
        if os.path.exists('confusion.mat'):
            print('Load confusion.mat!')
            cv=scipy.io.loadmat('confusion.mat')['cvconfusion']
            print('confusion matrix: \n {}'.format(cv))
        elif os.path.exists('xval_results.mat'):
            print('Load xval_results.mat!')
            cv=scipy.io.loadmat('xval_results.mat')['cvconfusion']
            cv = np.sum(cv,axis=2)
        else:
            print('No confusion!')
            exit()
    class_num=cv.shape[0]
    F1 = np.zeros((class_num,1))#4个类别各自F1
    for i in range(class_num):
        if (np.sum(cv[i,:])+np.sum(cv[:,i])) !=0:
            F1[i]=2*cv[i,i]/(np.sum(cv[i,:])+np.sum(cv[:,i]))
        else:
            F1[i]=0     
        print("F1 measure for {} rhythm: {:1.4f}".format(classes[i],F1[i,0]))
    # print(np.nonzero(F1[:,0]))
    F1mean = np.sum(F1)/len(np.nonzero(F1[:,0])[0])

    ################# generate result files 
    with open('result.txt','a') as f:
        #cal accuracy
        correct=0
        for i in range(class_num):
            correct+=cv[i,i]
        accuracy=correct/np.sum(cv) 
        f.write('\nTime :{} \n'.format(timenow))
        f.write('Confusion : \n{} \n'.format(cv))
        f.write('Accuracy(of all) : {:1.4f} \n'.format(accuracy))
        
        f.write('{:^15}{:^10}{:^10}{:^10}{:^10} \n'.format(' ',classes[0],classes[1],classes[2],classes[3]))
        #cal precision
        f.write('{:^15}'.format('precision: '))
        for i in range(class_num):
            if np.sum(cv[:,i])!=0:
                precision=cv[i,i]/np.sum(cv[:,i])
            else:
                precision=0
            f.write('{:^10.4f}'.format(precision))
        f.write('\n')
        #cal recall
        f.write('{:^15}'.format('recall: '))
        for i in range(class_num):
            if np.sum(cv[i,:])!=0:
                recall=cv[i,i]/np.sum(cv[i,:])
            else:
                recall=0
            f.write('{:^10.4f}'.format(recall))
        f.write('\n')
        #cal f1
        f.write('{:^15}'.format('F1: '))
        for i in range(class_num):
            f.write('{:^10.4f}'.format(F1[i][0]))
        f.write('\n F1mean(nonzero): {:^10.4f} \n'.format(F1mean))
    print('finish writing results!')

if Test:
    def main():
        ############# 正常情况
        # ytrue=[0,1,2,0,1,2,2,1,1,3]
        # ypred=[0,2,0,1,0,1,2,1,3,2]
        ############# 不正常正常情况1
        # ytrue=[0,1,2,0,1,2,2,1,1,3]
        # ypred=[0,1,0,1,0,1,1,1,1,1]
        ############# 不正常正常情况2
        ytrue=[0,1,3,0,1,1,1,3,3,1]
        ypred=[0,2,0,3,0,2,1,3,1,1]

        scores(ytrue,ypred)
else:
    def main():
        scores(None,None)

if __name__ == '__main__':
    main()