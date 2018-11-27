# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:48:16 2018

@author: Wei Wan
Deep learning with Python listing 5.13 -- listing 
"""

import os
import numpy as np

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# Path to the directory for orignal dataset and directory for storing for train
data_dir = 'C:/Users/Wei Wan/Documents\DeepLearning/DL with Python/data/train'
base_dir ='C:/Users/Wei Wan/Documents/DeepLearning/DL with Python'

# set directory for train data images
train_dir = os.path.join(base_dir, 'train')
# set directory for train cats images
train_cats_dir = os.path.join(train_dir,'cats')
# set directory for train dogs images
train_dogs_dir = os.path.join(train_dir,'dogs')

# set directory for validation images
validation_dir = os.path.join(base_dir, 'validation')
# set directory for validation cats images
validation_cats_dir = os.path.join(validation_dir,'cats')
# set directory for validation dogs images
validation_dogs_dir = os.path.join(validation_dir, 'dogs')


# set directory for test data images
test_dir = os.path.join(base_dir,'test')

# set directory for test cats images
test_cats_dir = os.path.join(test_dir, 'cats')

# set directory for test dogs images
test_dogs_dir = os.path.join(test_dir,'dogs')


#installing a small convnet for dogs vs. cats classification
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Configuring the model for training
model.compile(loss='binary_crossentropy', 
              optimizer = optimizers.RMSprop(lr=1e-4), 
              metrics=['accuracy'])

# Rescales all images by 1/255
train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        )

# validation data shouldn't be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,              # target directory
        target_size=(150,150),  # resizes all images to 150 X 150
        batch_size=25,
        class_mode='binary')    # Because use binary_crossentropy loss,

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=25,
        class_mode='binary')

# Fitting the model using a batch generator
history =  model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
#history=model.fit_generator(train_generator, steps_per_epoch=100, epochs-30,
#                            validation_data=validation_generator,
#                            validation_steps=50)

# Saving the model
model.save('cats_and_dogs_small_2.h5')

# Listing5.10 Displaying curves of loss and accuracy during training
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

# Listing 5.16 instantlating the VGG16 convultyuional base
from keras.applications import VGG16
conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape=(150,150,3))
model.summary()

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
