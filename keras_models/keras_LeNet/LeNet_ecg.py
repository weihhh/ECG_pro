#coding=utf-8  
import keras  
from keras.models import Sequential  
from keras.layers import Dense,Flatten  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical#one_hot
from keras.callbacks import TensorBoard

import numpy as np  

# seed = 7  
# np.random.seed(seed)  

import ecg_input  

CLASSES=5
matrix_h=300
matrix_w=1

(X_train, y_train), (X_test, y_test)=ecg_input.get_data()

X_train=X_train.reshape((X_train.shape[0],matrix_h,matrix_w,1))
X_test=X_test.reshape((X_test.shape[0],matrix_h,matrix_w,1))

model = Sequential()  
model.add(Conv2D(32,(5,1),strides=(1,1),input_shape=(matrix_h,matrix_w,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,1)))  
model.add(Conv2D(64,(5,1),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,1)))  
model.add(Flatten())  
model.add(Dense(100,activation='relu'))  
model.add(Dense(CLASSES,activation='softmax'))  
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])  
model.summary()  
  
model.fit(X_train,y_train,batch_size=2,epochs=2,verbose=2,validation_split=0.2,callbacks=[TensorBoard(log_dir='./tmp/log')])  
#[0.031825309940411217, 0.98979999780654904] 
score=model.evaluate(X_test,y_test,batch_size=20,verbose=2) 
print('Test loss:', score[0])  
print('Test accuracy:', score[1])  