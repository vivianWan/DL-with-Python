# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:44:56 2018
Deep Learning with Python

DL for computer vision
listing5.1, listing5.2 listing5.3

@author: Wei Wan
"""

from keras import models
from keras import layers
from keras.datasets import  mnist
from keras.utils import to_categorical

#Listing 5.1  Define the model as a stack of ConvD and MaxPooling2D layers.
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation ='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.summary()

# Listing 5.2 Adding a calssifier on top of the convnet
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# Listing 5.3 Training the convent on MNIST Images
# loading data
(train_images, train_labels), (test_images,test_labels) = mnist.load_data()

train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float') / 255

test_images = test_images.reshape((10000, 28,28,1))
test_images = test_images.astype('float')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer = 'rmsprop',
              loss='categorical_crossentropy',
              metrics =['accuracy'])
model.fit(train_images, train_labels, epochs = 5, batch_size=64)