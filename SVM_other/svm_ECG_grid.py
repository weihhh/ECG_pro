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
from sklearn.metrics import classification_report,accuracy_score #模型准确率,查准率，查全率,f1_score
from sklearn.model_selection import GridSearchCV#用于调参


test_clf=False#true表示仅测试分类器，无需训练

#重定向输出
output_file='out.txt'
print('输出重定向到{}'.format(output_file))
with open(output_file, 'w+') as file:
    sys.stdout = file  #标准输出重定向至文件

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
        #刷新缓冲区
        file.flush()
        #设置调参范围
        tuned_parameters=[{'kernel': ['rbf'], 'gamma': [0.001,0.01],
                         'C': [0.1,1,10]},
                        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
        start_time=time.time()
        clf=GridSearchCV(svm.SVC(),tuned_parameters,cv=4,verbose=5)
        clf.fit(x_train,y_train)
        with open('svmclf.pickle','wb') as f:
            pickle.dump(clf,f)
        print('最佳参数：{}，最佳分数：{}'.format(clf.best_params_,clf.best_score_))
        print('完成训练： {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ))
        print('训练用时： {}'.format(time.time()-start_time))

        #刷新缓冲区
        file.flush()

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


