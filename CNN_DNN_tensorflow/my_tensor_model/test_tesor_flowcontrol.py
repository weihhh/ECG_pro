#test_tesor_flowcontrol

import tensorflow as tf
import numpy as np
ypred=tf.placeholder(shape=(2,),dtype=tf.int32,name='ypred')
labels=tf.placeholder(shape=(2,),dtype=tf.int32,name='labels')

a=tf.constant([[1],[2],[3]]) 
b=tf.constant([[4],[5],[6]])
# d=tf.constant([[0,0],[0,0],[0,0]])
print(a.shape)
c = tf.stack([a,b],axis=1) 
# e=tf.add(c,d)



x=tf.constant(1)
# y = tf.cond(ypred,lambda:x+1,lambda:x-1)
def for_where():
    a=list()
    global x
    # for i in range(8):
    #     x+=1
    # for i in range(2):
    z1=tf.where(tf.equal(labels[0],5),2,0)
    x+=z1
    return x
z=for_where()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    c_stack=sess.run(c)
    z1,labels1=sess.run([z,labels],feed_dict={ypred:np.array([1,2]),labels:np.array([5,1])})
    z2,labels2=sess.run([z,labels],feed_dict={ypred:np.array([1,2]),labels:np.array([6,5])})
    print(z1,labels1,z2,labels2)
    print('c_stack',c_stack)
# import tensorflow as tf
# pred=tf.placeholder(dtype=tf.bool,name='bool')
# x=tf.constant(1)
# y = tf.cond(pred,lambda:x+1,lambda:x-1)
# z=tf.where(pred,x+1,x-1)

# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())
#     y1,z1=sess.run([y,z],feed_dict={pred:True})
#     y2,z2=sess.run([y,z],feed_dict={pred:False})
#     print(y1,z1,y2,z2)