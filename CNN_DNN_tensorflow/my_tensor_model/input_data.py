from sklearn.model_selection import train_test_split#训练数据、测试数据切分
from collections import Counter
import pickle
from sklearn import preprocessing
from sklearn.metrics import classification_report,accuracy_score #模型准确率,查准率，查全率,f1_score

from tensor_ecg_model import tensor_dataset


def read_data_sets():

    #unpickle
    with open(r'D:\aa_work\ECG\svm\data.pickle','rb') as f:
        ECG_data,ECG_annotation=pickle.load(f)#array of array,(*, 300),(*,1) *样本数
    print('原始数据规模： ',ECG_data.shape,'原始标签规模： ',ECG_annotation.shape)

    annotation_counts=Counter(ECG_annotation.flatten())
    print('类别概览： ',annotation_counts)

    #归一化
    ECG_data=preprocessing.scale(ECG_data)

    x_train,x_validation,y_train,y_validation=train_test_split(ECG_data,ECG_annotation.flatten(),test_size=0.5)
    print('训练集规模： {}，测试集规模： {}'.format(x_train.shape,x_validation.shape))


    train_dataset=tensor_dataset(x_train,y_train)
    validation_dataset=tensor_dataset(x_validation,y_validation)
    return train_dataset,validation_dataset


def main():
    train_dataset,validation_dataset=read_data_sets()

    print('训练集：{}，{}'.format(train_dataset._images.shape,train_dataset._labels.shape))
    print('验证集：{}, {}'.format(validation_dataset._images.shape,validation_dataset._labels.shape))
if __name__ == '__main__':
  main()

