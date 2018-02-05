from sklearn.model_selection import train_test_split#训练数据、测试数据切分
from collections import Counter
import pickle
from sklearn import preprocessing
import numpy as np

from Alexnet_ecg_model import tensor_dataset


def read_data_sets():

    #unpickle
    with open(r'D:\aa_work\ECG\ECG_DATA_ALL\ECG_10segment_data_sum.pickle','rb') as f:
        all_ecg_list,all_ann_list=pickle.load(f)
    
    print('所有心电信号：{} '.format(np.array(all_ecg_list).shape))#(10758, 2880) 
    print('所有心电信号标注：{} '.format(np.array(all_ann_list).shape))#(10758, 10)

    all_counts=Counter(np.array(all_ann_list).flatten())
    #字典无序，这里进行了排序显示
    print('类别概览： ',sorted(all_counts.items(),key=lambda x:x[0]))

    #axis等于0，表示拆开第一维相接：这里将所有文件拼接为整个，不考虑病人间的泛化
    all_spike_matrix=np.array(all_ecg_list)
    print('all_spike_matrix : ',all_spike_matrix.shape)#(10758, 2880)
    all_ann_matrix=np.array(all_ann_list)
    print('all_ann_matrix : ',all_ann_matrix.shape)

    #归一化,axis=0表示对列特征进行归一化，而axis=1表示对样本行进行
    ECG_data=preprocessing.scale(all_spike_matrix,axis=1)

    x_train,x_validation,y_train,y_validation=train_test_split(ECG_data,all_ann_matrix,test_size=0.5)
    # print('训练集规模： {}、{}，测试集规模： {}、{}'.format(x_train.shape,y_train.shape,x_validation.shape,y_validation.shape))


    train_dataset=tensor_dataset(x_train,y_train)
    validation_dataset=tensor_dataset(x_validation,y_validation)
    return train_dataset,validation_dataset


def main():
    train_dataset,validation_dataset=read_data_sets()

    print('训练集：{}，{}'.format(train_dataset._images.shape,train_dataset._labels.shape))
    print('验证集：{}, {}'.format(validation_dataset._images.shape,validation_dataset._labels.shape))
if __name__ == '__main__':
  main()

