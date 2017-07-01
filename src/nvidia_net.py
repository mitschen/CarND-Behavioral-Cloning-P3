'''
Created on 01.07.2017

@author: micha
'''


import pickle
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import cv2
import os
import csv
import sklearn

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D
from keras.backend.tensorflow_backend import dropout
from keras.models import load_model
from sklearn.model_selection import train_test_split

from shutil import copyfile
import os
import glob
import matplotlib.pyplot as plt


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.cvtColor(cv2.imread(batch_sample), cv2.COLOR_BGR2RGB)
                head, filename = os.path.split(batch_sample)
                #TAKE CARE HOW MANY UNDERLINES WE'RE EXPECTING
                angle = float(filename.split("_")[2][:4]) / 1000.
                if "lm" in filename: angle -= 0.02
                elif "rm" in filename: angle += 0.02
                elif "l" in filename: angle += 0.02
                elif "r" in filename: angle -= 0.02
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    

def readData(filepath):
    samples = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples
    
def readFilesFromFolder(filepath):
    samples = glob.glob("{0:s}{1:s}".format(destpath, "*.jpg"))
    return samples    

def trainModel(filepath, resume = False):
    
    modelnameIdx = 0
    model = None
    if resume == True:
        modelnames = glob.glob("*.h5")
        for name in modelnames:
            val = int(name.split("_")[1][:3])
            if val > modelnameIdx: modelnameIdx = val
        
    modelname = ("model_{0:03d}.h5".format(modelnameIdx), "model_{0:03d}.h5".format(modelnameIdx+1))
    if modelname[0] == 1:
        resume = False
    
    #readin the data    
    samples = readFilesFromFolder(filepath)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    
    #either resume or start from the scratch
    if resume == True:
        model = load_model(modelname[0])
    else:
        input_shape = (160,320,3)
        
        #specify the model
        model = Sequential()
        #cropping takes a while on my machine - so preprocessing and store the
        #data somewhere on HDD should improve runtime
        cropTop = 80
        cropBottom = 32
        model.add(Cropping2D(cropping=((cropTop,cropBottom), (0,0)), input_shape=input_shape))
            
        input_shape = np.subtract(input_shape, (cropTop+cropBottom,0,0))
        print (input_shape)
        
        #preprocessing normalization
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = input_shape))
        
        
        #the nvidia model
        model.add(Convolution2D(24,5,5, activation='relu'))
        model.add(MaxPooling2D(strides=(1,2)))
        model.add(Convolution2D(36,5,5, activation='relu'))
        model.add(MaxPooling2D(strides=(1,2)))
        model.add(Convolution2D(48,5,5, activation='relu'))
        model.add(MaxPooling2D(strides=(1,2)))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(MaxPooling2D(strides=(2,2)))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(MaxPooling2D(strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
#         model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
#         model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
#         model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu')) # must be one steering angle at the end
        model.add(Dense(1))
        print (model.summary())
                  
    #specify the generators          
    trainGen = generator(train_samples, batch_size=256)
    validGen = generator(validation_samples, batch_size=256)
    
    model.compile(loss='mse', optimizer='adam')
    sampleSize = len(train_samples)
    epochNo = 3
    while sampleSize < 15000:
        epochNo += 1
        sampleSize +=sampleSize
    history_Obj = model.fit_generator(trainGen, samples_per_epoch=len(train_samples), validation_data=validGen, nb_val_samples=len(validation_samples), nb_epoch=epochNo, verbose = 1)
    
#     plt.plot(history_Obj.history['loss'])
#     plt.plot(history_Obj.history['val_loss'])
#     plt.title('model mean squared error loss')
#     plt.ylabel('mean squared error loss')
#     plt.xlabel('epoch')
#     plt.legend(['training set', 'validation set'], loc='upper right')
#     plt.show()
    
    model.save(modelname[1])

    

    
def augmentation(filepath, destpath):
    samples = readData(filepath)
    list = []
    no = 0
    #figure out the latest offset
    for filename in glob.glob("{0:s}{1:s}".format(destpath, "*.jpg")):
        val = int(filename[len(destpath):].split("_")[0])
        if no < val: no = val 
    no +=1
    filename = ""
    for line in samples:
        angle = int(round(float(line[3])*1000.))
        filename = "{0:s}{1:d}_{2:s}_{3:04d}.jpg".format(destpath, no, "c", angle)
        copyfile(line[0], filename)
        no+=1
        #the flipped one
        filename = "{0:s}{1:d}_{2:s}_{3:04d}.jpg".format(destpath, no, "cm", -angle)
        no+=1
        cv2.imwrite(filename, cv2.flip(cv2.imread(line[0]), 1), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
         
        filename = "{0:s}{1:d}_{2:s}_{3:04d}.jpg".format(destpath, no, "l", angle)
        copyfile(line[1], filename)
        no+=1
        #the flipped one
        filename = "{0:s}{1:d}_{2:s}_{3:04d}.jpg".format(destpath, no, "lm", -angle)
        no+=1
        cv2.imwrite(filename, cv2.flip(cv2.imread(line[1]), 1), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
         
        filename = "{0:s}{1:d}_{2:s}_{3:04d}.jpg".format(destpath, no, "r", angle) 
        copyfile(line[2], filename)
        no+=1
        #the flipped one
        filename = "{0:s}{1:d}_{2:s}_{3:04d}.jpg".format(destpath, no, "rm", -angle)
        no+=1
        cv2.imwrite(filename, cv2.flip(cv2.imread(line[2]), 1), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                

if __name__ == '__main__':
    filepath = '../IMG4/driving_log.csv'
    destpath = 'R:\\AUG_IMG\\'
    #augmentation(filepath, destpath)
    trainModel(destpath, False)
    pass