#ECG识别模型

import math

import tensorflow as tf
import numpy as np

#分类问题，30个输出
NUM_CLASSES = 5

# 96x96
IMAGE_SIZE_h = 1
IMAGE_SIZE_w=300
IMAGE_PIXELS = IMAGE_SIZE_h * IMAGE_SIZE_w
LABEL_SIZE=NUM_CLASSES


#卷积和池化,1步长（stride size），0边距（padding size）的模板,池化用简单传统的2x2大小的模板
def conv2d(x,W):#二维卷积
    '''
    W滤波器即权重参数
    '''
    return tf.nn.conv2d(x,W,strides=[1,1,3,1],padding='VALID')

#logits: 未归一化的概率， 一般也就是 softmax层的输入
def inference(images, conv1_chanel, conv2_chanel,full_chanel):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  
  images=tf.reshape(images,[-1,IMAGE_SIZE_h,IMAGE_SIZE_w,1])

  # 第一层卷积：1x5，1x3 32
  with tf.name_scope('conv1'):
    weights = tf.Variable(
        tf.truncated_normal([1,5,1,conv1_chanel],
                            stddev=0.1 ),
        name='weights')
    biases = tf.Variable(tf.zeros([conv1_chanel]),
                         name='biases')
    conv1 = tf.nn.relu(conv2d(images, weights) + biases)
    pool1=tf.nn.max_pool(conv1,ksize=[1,1,3,1],strides=[1,1,2,1],padding='VALID')

  # 第二层卷积：1x3，1x3 64
  with tf.name_scope('conv2'):
    weights = tf.Variable(
        tf.truncated_normal([1,3,conv1_chanel,conv2_chanel],
                            stddev=0.1),
        name='weights')
    biases = tf.Variable(tf.zeros([conv2_chanel]),
                         name='biases')
    conv2 = tf.nn.relu(conv2d(pool1, weights) + biases)
    pool2=tf.nn.max_pool(conv2,ksize=[1,1,3,1],strides=[1,1,2,1],padding='VALID')


  #平铺
  pool2=tf.reshape(pool2,[-1,64*7*1])
  # 全连接
  with tf.name_scope('full_layer'):
    weights = tf.Variable(
        tf.truncated_normal([64*7*1, full_chanel],
                            stddev=0.1),
        name='weights')
    biases = tf.Variable(tf.zeros([full_chanel]),
                         name='biases')
    full_out = tf.matmul(pool2, weights) + biases


  # 输出层
  with tf.name_scope('out_layer'):
    weights = tf.Variable(
        tf.truncated_normal([full_chanel, NUM_CLASSES],
                            stddev=0.1),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(full_out, weights) + biases

    # #增加logits处理，提升不正常波形判定几率
    # add_percent=tf.Variable(tf.ones([256, NUM_CLASSES]), name='add_percent')
    # def change_logits():
    #   for i in range(NUM_CLASSES):
    #     for j in range(256):
    #       if i ==0:
    #         # add_percent[j,i].assign(tf.negative(percent_10[j]))
    #         tf.assign(add_percent[j,i],tf.negative(percent_10[j]))
    #       else:
    #         # add_percent[j,i].assign(percent_10[j]/(NUM_CLASSES-1))
    #         tf.assign(add_percent[j,i],percent_10[j]/(NUM_CLASSES-1))
    #   logits_boost=logits+add_percent
    #   return logits_boost
    # percent_10=tf.divide(tf.reduce_sum(logits,1),10)
    # logits_boost=change_logits()

    # # add_percent=tf.stack([percent_10,percent_10,percent_10,percent_10,percent_10],axis=2)
    # # logits=tf.add(logits,add_percent)
  return logits

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss



def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  
  # Create the gradient descent optimizer with the given learning rate.
  
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def accuracy(logits,labels):
  num_examples=labels.shape[0]
  correct_bool = tf.nn.in_top_k(logits, labels, 1)
  correct_int=tf.cast(correct_bool, tf.int32)#对应标签反应是否预测正确
  correct=tf.reduce_sum(correct_int)
  precision = correct / num_examples
  return precision,correct_int,labels

class  tensor_dataset(object):
  """docstring for  tensor_dataset"""
  def __init__(self,images,labels):
    super( tensor_dataset, self).__init__()
    self._index_in_epoch=0
    self._epochs_completed=0
    #images即数据数组
    self._images=images
    self._labels=labels
    self._num_examples=images.shape[0]

  def next_batch(self,batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples: # epoch中的句子下标是否大于所有语料的个数，如果为True,开始新一轮的遍历
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples) # arange函数用于创建等差数组
      np.random.shuffle(perm)  # 打乱
      #会复制
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

