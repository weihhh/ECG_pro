import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split#训练数据、测试数据切分
from collections import Counter
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,accuracy_score #模型准确率,查准率，查全率,f1_score



'''
卷积神经网络
'''

#unpickle
with open(r'D:\aa_work\ECG\svm\data.pickle','rb') as f:
    ECG_data,ECG_annotation=pickle.load(f)#array of array,(*, 300),(*,1) *样本数
ECG_annotation=ECG_annotation.reshape(-1,1)
print('原始数据规模： ',ECG_data.shape,'原始标签规模： ',ECG_annotation.shape)

#查看数据分类
annotation_counts=Counter(ECG_annotation.flatten())
print('类别概览： ',annotation_counts)

#onehot
enc=OneHotEncoder()
enc.fit(ECG_annotation)#注意这里需要是行向量输入，若单标签，则reshape(-1,1)
ECG_annotation=enc.transform(ECG_annotation).toarray()
print('onehot处理后：{} '.format(ECG_annotation.shape))#x*5

#归一化
ECG_data=preprocessing.scale(ECG_data)




x = tf.placeholder("float", shape=[None, 300])#训练样本占位符
y_ = tf.placeholder("float", shape=[None, 5])#标签占位符

#权重初始化
'''
权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度,
使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题
'''
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)#根据stddev方差返回正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.01,shape=shape)
    return tf.Variable(initial)

#卷积和池化,1步长（stride size），0边距（padding size）的模板,池化用简单传统的2x2大小的模板
def conv2d(x,W):#二维卷积
    '''
    W滤波器即权重参数
    '''
    return tf.nn.conv2d(x,W,strides=[1,1,3,1],padding='VALID')

#第一层卷积层
W_conv1=weight_variable([1,5,1,32])#5x5patch中算出32个特征，前两个是维度大小filter_height,filter_width,in_channels,channel_multiplier，接着是输入的通道数目，最后是输出的通道数目。
b_conv1=bias_variable([32])#每个输出都有一个偏置

x_image=tf.reshape(x,[-1,1,300,1])#28x28为图片大小，1为通道数，因为这里是灰度图,-1表示让系统自动算出此维度，其他维度固定，则此维度唯一

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,1,3,1],strides=[1,1,2,1],padding='VALID')

print('第一层')

#第二层卷积层
W_conv2=weight_variable([1,3,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,1,3,1],strides=[1,1,2,1],padding='VALID')

print('第二层')


#密集连接层,1024个神经元全链接
w_fc1=weight_variable([1*7*64,256])
b_fc1=bias_variable([256])

h_pool2_flat=tf.reshape(h_pool2,[-1,1*7*64])#1，7*7*64
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)#产出的是一维向量

print('全连接')


#dropout，防止过拟合
keep_prob=tf.placeholder('float')
h_tfc_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([256, 5])
b_fc2 = bias_variable([5])

y_conv=tf.nn.softmax(tf.matmul(h_tfc_drop, W_fc2) + b_fc2)
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv+1e-10))

# y_conv=tf.matmul(h_tfc_drop, W_fc2) + b_fc2
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#       logits=y_conv, labels=y_, name='xentropy')
# cross_entropy = tf.reduce_mean(cross_entropy, name='xentropy_mean')

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

sess = tf.InteractiveSession()#交互式session，可以在图运行的时候插入操作
sess.run(tf.initialize_all_variables())


max_epoch=50#1个epoch就是将所有的训练图像全部通过网络训练一次 
itr=0#记录epoch的次数
batch_size=100

while itr<max_epoch:
    itr+=1
    print('epoch次数：{}'.format(itr))
    # batch=sess.run(tf.train.shuffle_batch([x_train,y_train],batch_size=256, capacity=1024, min_after_dequeue=300))
    #batch_size是返回的一个batch样本集的样本个数。capacity是队列中的容量,min_after_dequeue这个代表队列中的元素小于它的时候就补充
    
    #拆分训练和测试样本
    x_train,x_test,y_train,y_test=train_test_split(ECG_data,ECG_annotation,test_size=0.2)
    
    print('训练集规模： {}，测试集规模： {}'.format(x_train.shape,x_test.shape))
    
    for batch_i in range(len(x_train)//batch_size):
        batch=x_train[batch_i*batch_size:(batch_i+1)*batch_size]
        label_batch=y_train[batch_i*batch_size:(batch_i+1)*batch_size]
        
        train_accuracy=accuracy.eval(feed_dict={x:batch,y_:label_batch,keep_prob:1})
        train_loss=cross_entropy.eval(feed_dict={x:batch,y_:label_batch,keep_prob:1})
        if itr%2==0 and batch_i%20==0:
            print('loss: {}'.format(train_loss))    
            print('step{} ,training accuracy {}'.format(batch_i,train_accuracy))
        train_step.run(feed_dict={x:batch,y_:label_batch,keep_prob:0.5})
print('test accuracy: {}'.format(accuracy.eval(feed_dict={x:x_test,y_:y_test,keep_prob:1})))






