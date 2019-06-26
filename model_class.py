

import pandas as pd
data=pd.read_csv('/Users/saumyakansal/Desktop/SAUMYA/Python and ML/Devdata.csv')
print(data[0:5])
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K

dataset=np.array(data)
np.random.shuffle(dataset)
X=dataset
Y=dataset
X=X[:,0:1024]
Y=Y[:,1024]
x_train=X[0:70000,:]
x_train=x_train/255
x_test=X[70000:72001,:]
x_test=x_test/255

Y=Y.reshape(Y.shape[0],1)
y_train=Y[0:70000,:]
y_train=y_train.T
y_test=Y[70000:72001]
y_test=y_test.T

print(X[0:5])

print(x_train.shape[0])
print(x_test.shape[0])
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

image_x=image_y=32
train_y=np_utils.to_categorical(y_train)
test_y=np_utils.to_categorical(y_test)
print(train_y.shape)
train_y=train_y.reshape(train_y.shape[1],train_y.shape[2])
test_y=test_y.reshape(test_y.shape[1],test_y.shape[2])

x_train=x_train.reshape(x_train.shape[0],image_x,image_y,1)
x_test=x_test.reshape(x_test.shape[0],image_x,image_y,1)

print(x_train.shape,train_y.shape,test_y.shape,x_test.shape)

def k_model(image_x,image_y):
  n_class=37
  model=Sequential()
  model.add(Conv2D(filters=32, kernel_size=(5,5),
                   input_shape=(image_x,image_y,1),
                   activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
  model.add(Conv2D(64,(5,5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(5,5), strides=(5,5), padding='same'))
  model.add(Flatten())
  model.add(Dense(n_class, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam',
               metrics=['accuracy'])
  filepath='devnagri.h5'
  cp1=ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                      save_best_only=True, mode='max')
  callback_list=[cp1]
  
  return model,callback_list

model, callback_list=k_model(image_x,image_y)
model.fit(x_train,train_y, validation_data=(x_test,test_y), epochs=20,
         batch_size=64, callbacks=callback_list)
scores=model.evaluate(x_test,test_y,verbose=0)
print('CNN error: %.2f%%' %(100-scores[1]*100))
print_summary(model)
model.save('devnagri.h5')