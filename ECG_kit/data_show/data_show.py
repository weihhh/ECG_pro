import numpy as np      #科学计算包 
import matplotlib.pyplot as plt      #python画图包   
import pickle
from sklearn import preprocessing
from collections import Counter

#注意显示的pickle文件！！！！！！！！

#unpickle
with open(r'D:\aa_work\ECG\svm\all_type_data.pickle','rb') as f:
    ECG_data,ECG_annotation=pickle.load(f)#array of array,(*, 300),(*,1) *样本数
print('原始数据规模： ',ECG_data.shape,'原始标签规模： ',ECG_annotation.shape)

annotation_counts=Counter(ECG_annotation.flatten())
print('类别概览： ',annotation_counts)


for i,data in enumerate(ECG_data[90000:100000]):
    plt.figure(i)
    plt.plot(data)
    plt.show()

