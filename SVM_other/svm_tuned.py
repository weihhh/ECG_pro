import numpy as np      #科学计算包  
import pickle
import time
import sys
import matplotlib.pyplot as plt      #python画图包  
from collections import Counter
#机器学习库
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split#训练数据、测试数据切分
from sklearn.metrics import classification_report,accuracy_score,f1_score #模型准确率,查准率，查全率,f1_score




#unpickle
with open(r'D:\aa_work\ECG\svm\data.pickle','rb') as f:
    ECG_data,ECG_annotation=pickle.load(f)#array of array,(*, 300),(*,1) *样本数
print('原始数据规模： ',ECG_data.shape,'原始标签规模： ',ECG_annotation.shape)
annotation_counts=Counter(ECG_annotation.flatten())
print('类别概览： ',annotation_counts)

#归一化
# ECG_data=preprocessing.scale(ECG_data)

x_train,x_test,y_train,y_test=train_test_split(ECG_data,ECG_annotation.flatten(),test_size=0.5)
print('训练集规模： {}，测试集规模： {}'.format(x_train.shape,x_test.shape))


print('开始训练： {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ))
def svm_fun(args):
    print('one')
    start_time=time.time()
    clf=svm.SVC(C=args['C'],gamma=args['gamma'])
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    report=classification_report(y_test,y_pred,digits=4)
    F1=f1_score(y_test, y_pred, average='weighted')
    with open('svm_fun_report.txt','a') as f:
        f.write('TIME:{}---C:{},gamma:{} \n report:{} \n F1:{} \n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            args["C"],args["gamma"],report,str(F1)))
    return -F1

if __name__ == '__main__':
    svm_fun({'C':0.1,'gamma':0.01})
