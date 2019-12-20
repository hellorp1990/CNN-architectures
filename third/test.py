import numpy as np
np.random.seed(2016)
import scipy
import os
import glob
import math
import pickle
import datetime
#import pandas as pd

from keras.models import Model
from keras.layers.advanced_activations import ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.layers.core import  ActivityRegularization
from keras.optimizers import RMSprop
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers import Merge
from keras.regularizers import l2
# dimensions of our images.
img_width, img_height = 100,100

train_data_dir = '/home/rahulp/keras-1/train'
validation_data_dir = '/home/rahulp/keras-1/validation'
test_data_dir = '/home/rahulp/Desktop/test'
nb_train_samples = 13536
nb_validation_samples = 5256
nb_test_samples = 237

nb_epoch = 100
input_layer = Reshape((100,100,3), input_shape=(img_width, img_height,3))
input_layer2 = Reshape((10,10,3), input_shape=(img_width, img_height,3))

leftBranch = Sequential()
leftBranch.add(input_layer)
#leftBranch.add(Convolution2D(64, 11, 11))
leftBranch.add(LeakyReLU(alpha=.01))
leftBranch.add(MaxPooling2D(pool_size=(10, 10)))
leftBranch.add(Dropout(0.1))
#leftBranch.add(Flatten())

rightBranch = Sequential()
rightBranch.add(input_layer)
rightBranch.add(Convolution2D(64, 5, 5))
rightBranch.add(LeakyReLU(alpha=.01))
rightBranch.add(MaxPooling2D(pool_size=(3, 3)))

#rightBranch.add(Convolution2D(30, 3, 3))
#rightBranch.add(LeakyReLU(alpha=.01))
#rightBranch.add(MaxPooling2D(pool_size=(2, 2)))

rightBranch.add(Convolution2D(64,2, 2 ))
rightBranch.add(LeakyReLU(alpha=.01))
rightBranch.add(MaxPooling2D(pool_size=(3, 3)))
rightBranch.add(Dropout(0.1))

#rightBranch.add(Flatten())

merged = Merge([leftBranch, rightBranch], mode='concat')

model = Sequential()

model.add(merged)
model.add(Convolution2D(66, 2, 2))
#model.add(PReLU())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
#model.add(Convolution2D(64, 1, 1))
model.add(Flatten(name='flat_1'))

model.add(ActivityRegularization(l2=0.01))
#model.add(Activation('relu'))


model.add(Dropout(0.1))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.load_weights("weights-improvement-04-0.81.hdf5")

rms =RMSprop( lr=0.0001)
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

np.savetxt("name.txt",test_generator.filenames, delimiter=" ", fmt="%s")

scores = model.evaluate_generator(test_generator, 237)
print("Accuracy = ", scores[1])

layer_name = 'flat_1'
intermediate_layer_model = Model(model.input,model.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict_generator(test_generator,len(test_generator.filenames))

np.savetxt("encoded.csv",intermediate_output, delimiter=",")

weights, bias = model.layers[-1].get_weights()



