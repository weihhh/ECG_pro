import numpy as np      #科学计算包  
import pickle
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split#训练数据、测试数据切分
from sklearn.metrics import classification_report,accuracy_score #模型准确率,查准率，查全率,f1_score
from collections import Counter
import time
import matplotlib.pyplot as plt      #python画图包  

test_clf=False#true表示仅测试分类器，无需训练


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


if test_clf:
    with open('svmclf.pickle','rb') as f:
        clf=pickle.load(f)
else:
    print('开始训练： {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ))
    start_time=time.time()
    clf=svm.SVC(C=10,gamma=0.01)
    clf.fit(x_train,y_train)
    print('完成训练： {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ))
    print('训练用时： {}'.format(time.time()-start_time))

    with open('svmclf.pickle','wb') as f:
        pickle.dump(clf,f)

#预测
y_pred=clf.predict(x_test)

#用于人工观察预测结果,单个进行预测
'''
for i in range(100):
    print('真实：{}--->预测：{}'.format(y_test[i],clf.predict([x_test[i]])))
    time.sleep(2)
'''

#用于打印某一类的全部测试集与其中被成功预测的波形
'''
plt.figure(1)
for i,test in enumerate(x_test):
        if y_test[i]==3:
            plt.plot(test)

plt.show()
'''


print('全类型总准确率： {}'.format(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred,digits=4)
print('综合报告： ',report)


