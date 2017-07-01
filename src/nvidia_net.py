'''
Created on 01.07.2017

@author: micha
'''


import pickle
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D


def trainModel(enableCropping):
    input_shape = (160,320,3)
    model = Sequential()
    
    #cropping takes a while on my machine - so preprocessing and store the
    #data somewhere on HDD should improve runtime
    if(True == enableCropping):
        model.add(Cropping2D(cropping=(50,20), (0,0)), input_shape=input_shape)
    else:
        input_shape = (50,20,3)
    
    #preprocessing normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = input_shape))
    
    
    #the nvidia model
    model.add(Convolution2D(24,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(36,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10)) # must be one steering angle at the end
              
              
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
    model.save('model.h5')

if __name__ == '__main__':
    pass