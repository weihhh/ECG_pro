from tensorflow.examples.tutorials.mnist import input_data
from my_tensor_model import tensor_dataset

dir=r'D:\aa_work\mnist'

def read_data_sets(dir):

    mnist=input_data.read_data_sets(dir)
    #构建返回值
    train_dataset=tensor_dataset(mnist.train.images,mnist.train.labels)
    validation_dataset=tensor_dataset(mnist.validation.images,mnist.validation.labels)
    test_dataset=tensor_dataset(mnist.test.images,mnist.test.labels)
    return train_dataset,validation_dataset,test_dataset



def main():
    train_dataset,validation_dataset,test_dataset=read_data_sets(dir)

    print('训练集：{}，{}'.format(train_dataset._images.shape,train_dataset._labels.shape))
    print('验证集：{}，{}'.format(validation_dataset._images.shape,validation_dataset._labels.shape))
    print('测试集：{}，{}'.format(test_dataset._images.shape,test_dataset._labels.shape))
if __name__ == '__main__':
  main()