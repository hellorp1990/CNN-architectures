import numpy as np
np.random.seed(2016)
import scipy
import os
import glob
import math
import pickle
import datetime
#import pandas as pd


from keras.layers.advanced_activations import ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,AveragePooling2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import  ActivityRegularization
from keras.optimizers import RMSprop


# dimensions of our images.
img_width, img_height = 100,100

train_data_dir = '/home/rahulp/keras-1/train'
validation_data_dir = '/home/rahulp/keras-1/validation'
test_data_dir = '/home/rahulp/keras-1/test/test'
nb_train_samples = 13536
nb_validation_samples = 5256
nb_test_samples = 237

nb_epoch = 100


model = Sequential()

model.add(Convolution2D(64, 5, 5, input_shape=(img_width, img_height,3)))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=.01))
#model.add(ELU())
model.add(MaxPooling2D(pool_size=(3, 3)))


model.add(Convolution2D(64, 2, 2))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=.01))
#model.add(ELU())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(ActivityRegularization(l2=0.01))

model.add(Activation('relu'))

model.add(Dropout(0.25))


model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights("weights-improvement-00-0.81.hdf5")

rms =RMSprop( lr=0.00001)
model.compile(loss='binary_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])


# this is the augmentation configuration we will use for training



test_datagen = ImageDataGenerator(rescale=1./255)


test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='binary')



scores = model.evaluate_generator(test_generator, 237)
print("Accuracy = ", scores[1])



