import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split#训练数据、测试数据切分
from collections import Counter
import pickle
from sklearn import preprocessing


#unpickle
with open(r'D:\aa_work\新学期项目及论文\心电图项目\svm\data.pickle','rb') as f:
    ECG_data,ECG_annotation=pickle.load(f)#array of array,(*, 300),(*,1) *样本数
print('原始数据规模： ',ECG_data.shape,'原始标签规模： ',ECG_annotation.shape)

annotation_counts=Counter(ECG_annotation.flatten())
print('类别概览： ',annotation_counts)

#归一化
ECG_data=preprocessing.scale(ECG_data)

x_train,x_test,y_train,y_test=train_test_split(ECG_data,ECG_annotation.flatten(),test_size=0.5)
print('训练集规模： {}，测试集规模： {}'.format(x_train.shape,x_test.shape))

feature_columns=[tf.contrib.layers.real_valued_column('',dimension=300)]

#声明特征取值
classifier=tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[1000,500,1000],n_classes=5,model_dir=r'D:\aa_work\model')

classifier.fit(x=x_train,y=y_train,steps=2000)

accuracy_score = classifier.evaluate(x=x_test,
                                     y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# 直接创建数据来进行预测
# new_samples = np.array(
#     [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
# y = classifier.predict(new_samples)
# print('Predictions: {}'.format(list(y)))