#ECG识别模型

import math

import tensorflow as tf
import numpy as np

#分类问题，30个输出
NUM_CLASSES = 60
TYPE_USED=6

# 96x96
IMAGE_SIZE_h = 1
IMAGE_SIZE_w=2880
IMAGE_PIXELS = IMAGE_SIZE_h * IMAGE_SIZE_w
LABEL_SIZE=NUM_CLASSES


#卷积和池化,1步长（stride size），0边距（padding size）的模板,池化用简单传统的2x2大小的模板
def conv2d(x,W):#二维卷积
    '''
    W滤波器即权重参数
    '''
    return tf.nn.conv2d(x,W,strides=[1,1,3,1],padding='VALID')

#logits: 未归一化的概率， 一般也就是 softmax层的输入
def inference(images):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  
  images=tf.reshape(images,[-1,IMAGE_SIZE_h,IMAGE_SIZE_w,1])

  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 11, 1, 96], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 1, 4, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)

    
  # lrn1
  with tf.name_scope('lrn1') as scope:
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool1
  pool1 = tf.nn.max_pool(lrn1,
                         ksize=[1, 1, 3, 1],
                         strides=[1, 1, 2, 1],
                         padding='VALID',
                         name='pool1')

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 5, 96, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)


  # lrn2
  with tf.name_scope('lrn2') as scope:
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool2
  pool2 = tf.nn.max_pool(lrn2,
                         ksize=[1, 1, 3, 1],
                         strides=[1, 1, 2, 1],
                         padding='VALID',
                         name='pool2')

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 3, 256, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 3, 384, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)


  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)


  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 1, 3, 1],
                         strides=[1, 1, 2, 1],
                         padding='VALID',
                         name='pool5')

  #平铺
  pool5=tf.reshape(pool5,[-1,256*84*1])
  # 全连接
  with tf.name_scope('full_layer'):
    weights = tf.Variable(
        tf.truncated_normal([256*84*1, 1000],
                            stddev=0.1),
        name='weights')
    biases = tf.Variable(tf.zeros([1000]),
                         name='biases')
    full_out = tf.matmul(pool5, weights) + biases

  #增加dropout层
  with tf.name_scope('full_layer'):
    # keep_prob=tf.placeholder('float')
    h_tfc_drop=tf.nn.dropout(full_out,0.5)

  # 输出层
  with tf.name_scope('out_layer'):
    weights = tf.Variable(
        tf.truncated_normal([1000, NUM_CLASSES],
                            stddev=0.1),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(h_tfc_drop, weights) + biases
  return logits

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  new_list=[]
  new_label_list=[]
  for i,j in  zip(range(0,NUM_CLASSES,TYPE_USED),range(0,10)):
      new_label_list.append(tf.slice(labels,[0,j],[-1,1]))
      new_list.append(tf.slice(logits,[0,i],[-1,TYPE_USED]))
  new=tf.concat(new_list,axis=0)

  new_label=tf.reshape(tf.concat(new_label_list,axis=0),[-1])

  new_label_onehot=tf.one_hot(new_label,TYPE_USED)
  y=tf.nn.softmax(new)  
  result_where=tf.where(tf.equal(tf.argmax(new_label_onehot,1),0),y*1.2+1e-10,y*0.8+1e-10)
  loss = -tf.reduce_mean(new_label_onehot*tf.log(result_where), name='xentropy_mean')  
  # cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=new, labels=new_label, name='xentropy')

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
  
  
  # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  learning_rate_decay = tf.train.exponential_decay(learning_rate, global_step, 100, 0.90, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def accuracy(logits,labels):
  #new_list存储分割后的数据，new_label_list存储分割后的标签
  new_list=[]
  new_label_list=[]

  for i,j in  zip(range(0,60,6),range(0,10)):
    #slice中的-1代表到底，这里slice相当于labels[0:][j:j+1]
    new_label_list.append(tf.slice(labels,[0,j],[-1,1]))
    new_list.append(tf.slice(logits,[0,i],[-1,6]))
  new=tf.concat(new_list,axis=0)
  new_label=tf.reshape(tf.concat(new_label_list,axis=0),[-1])

  pre_label=tf.argmax(new,1)

  correct_bool = tf.nn.in_top_k(new,new_label, 1)
  correct_int=tf.cast(correct_bool, tf.int32)#对应标签反应是否预测正确
  correct=tf.reduce_sum(correct_int)
  precision = correct /tf.shape(correct_int)[0]
  return precision,pre_label,new_label,labels,tf.shape(correct_int)[0]

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

