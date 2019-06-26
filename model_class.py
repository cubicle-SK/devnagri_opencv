

import pandas as pd
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('/Users/saumyakansal/Desktop/SAUMYA/Python and ML/Devdata.csv')
#print(data[0:5])
#print(data.groupby('character').count())
X=data.values[:,:-1]/255.0
Y=data['character'].values
del data #to minimize memory consumption
n_class=36
#print(X.shape,Y.shape)
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,
                                                random_state=42)
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)
y_train=np_utils.to_categorical(y_train, n_class)
y_test=np_utils.to_categorical(y_test, n_class)
#print(y_test[0:10])
x_train=x_train.reshape(x_train.shape[0],32,32,1)
x_test=x_test.reshape(x_test.shape[0],32,32,1)
#print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
'''
checking if correct data
plt.imshow(x_test[150])
plt.show()
print(test_y[150])
pred_class=list(test_y[150]).index(max(test_y[150]))
print(pred_class)
'''

def k_model():
  n_class=36
  model=Sequential()
  model.add(Conv2D(filters=32, kernel_size=(3,3),
                   input_shape=(32,32,1),
                   activation='relu'))
  model.add(Conv2D(filters=64, kernel_size=(3,3),
                   activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
  model.add(Conv2D(64,(3,3), activation='relu'))
  model.add(Conv2D(64,(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(units=128, activation='relu',
                  kernel_initializer='uniform'))
  model.add(Dense(units=64, activation='relu',
                  kernel_initializer='uniform'))
  model.add(Dense(n_class, activation='softmax',
                  kernel_initializer='uniform'))
  model.compile(loss='categorical_crossentropy', optimizer='adam',
               metrics=['accuracy'])
  filepath='devnagri.h5'
  #print(model.summary())
  cp1=ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                      save_best_only=True, mode='max')
  callback_list=[cp1]
  
  return model,callback_list

model, callback_list=k_model()
model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=1,
         batch_size=64, callbacks=callback_list)
scores=model.evaluate(x_test,y_test,verbose=0)
print('CNN error: %.2f%%' %(100-scores[1]*100))
print_summary(model)
model.save('devnagri.h5')