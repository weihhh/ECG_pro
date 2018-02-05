import numpy as np      #科学计算包  
import pickle
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split#训练数据、测试数据切分
from sklearn.metrics import classification_report #模型准确率,查准率，查全率,f1_score
from collections import Counter
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor 


#unpickle
with open(r'D:\aa_work\新学期项目及论文\心电图项目\svm\data.pickle','rb') as f:
    ECG_data,ECG_annotation=pickle.load(f)#array of array,(*, 300),(*,1) *样本数
print('原始数据规模： ',ECG_data.shape,'原始标签规模： ',ECG_annotation.shape)

annotation_counts=Counter(ECG_annotation.flatten())
print('类别概览： ',annotation_counts)
#归一化
# ECG_data=preprocessing.scale(ECG_data)

x_train,x_test,y_train,y_test=train_test_split(ECG_data,ECG_annotation.flatten(),test_size=0.5)
print('训练集规模： {}，测试集规模： {}'.format(x_train.shape,x_test.shape))

rf=RandomForestRegressor(n_estimators=5)#这里使用了默认的参数设置  
rf.fit(x_train,y_train)#进行模型的训练  
y_pred=rf.predict(x_test)
print(type(y_pred[0]),type(y_pred[0]))

with open('clf.pickle','wb') as f:
    pickle.dump(rf,f)


report=classification_report(y_test,y_pred)
# right=0
# for i,pre_class in enumerate(y_pred):
#     if pre_class==y_test[i]:
#         right+=1

# print('综合报告： ',right/len(y_pred))


