#test CNN segment

import tensorflow as tf
import numpy as np

#4*6
logits=np.array([[1,2,3,1,2,3],[2,3,4,1,2,3],[5,6,7,1,2,3],[8,9,10,1,2,3]],dtype=np.float32)
labels=np.array([[1,2],[0,2],[1,0],[0,1]],dtype=np.int32)

def loss_make(logits,labels):
    new_list=[]
    new_label_list=[]
    for i,j in  zip(range(0,6,3),range(0,2)):
        new_label_list.append(tf.slice(labels,[0,j],[-1,1]))
        new_list.append(tf.slice(logits,[0,i],[-1,3]))
    new=tf.concat(new_list,axis=0)
    new_label=tf.reshape(tf.concat(new_label_list,axis=0),[-1])
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=new, labels=new_label, name='xentropy')
    # for i,j in  zip(range(0,6,3),range(0,2)):
    #     cross_entropy= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:,i:i+3], labels=labels[:,j], name='xentropy')
    #     loss += tf.reduce_sum(cross_entropy, name='xentropy_mean')
    #     print('ok')
    return cross_entropy,new,new_label

with tf.Session() as sess:  
    cross_entropy,new,new_label=loss_make(logits,labels)
    
    # cross_entropy2= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:,i:i+3], labels=labels[:,j], name='xentropy')
    s,new_array,new_array1 = sess.run([cross_entropy,new,new_label])
    s2 = sess.run(cross_entropy)
    print('s2',s) 
    print('s2: ',s2) 
    print(new_array,'\n',new_array1)