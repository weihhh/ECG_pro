import numpy as np  

from keras.utils.np_utils import to_categorical#one_hot
from keras.datasets import mnist

def check_ndarray(x):
    print('shape: {}'.format(x.shape))
    print('size: {}'.format(x.size))
    print('type: {}'.format(x.dtype))
    print('itemsize(byte): {}'.format(x.itemsize))

def get_data():
    '''
    读取mnist数据,onehot
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data(path='mnist')
    #one_hot
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)

    X_train = X_train.astype('float32')  
    X_test = X_test.astype('float32')
    #关键！！！
    X_train/=255
    X_test/=255

    print('训练集数据：')
    check_ndarray(X_train)
    print('训练集标签：')
    check_ndarray(y_train)
    print('测试集数据：')
    check_ndarray(X_test)
    print('测试集标签：')
    check_ndarray(y_test)
    return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test)=get_data()
    print(y_train[0])