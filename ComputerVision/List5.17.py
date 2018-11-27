# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:44:54 2018
DL with Python
listing 5.17 extracting features using the pretrained convolutional base


Have not tested yet


@author: Wei Wan
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

base_dir = 'C:/Users/Wei Wan/Documents\DeepLearning/DL with Python/'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')

datagen=ImageDataGenerator(rescale=1./255)
batch_size=20

def extract_features(directory, sample_count):
    features=np.zeros(shape=(sample_count, 4,4,512))
    labels = np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(
            directory,
            target_size=(150,150),
            batch_size=batch_size,
            class_mode='binary')
    
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size: (i+1)*batch_size] = features_batch
        labels[i*batch_size: (i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size >= sample_count:
            break
        
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels= extract_features(test_dir, 1000)

# listing5.18 Defining and training the densely connected classifier
model = models.Sequential()
model.add(layers.Densen(256,activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape=(150,150,3))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels, epochs =30,
                    batch_size=0, validation_data=(validation_features, validation_labels))

acc = history.history['acc']
val_acc=history.history['val_acc']
loss = history.history['loss']
val_loss=history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
