# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:48:16 2018

@author: Wei Wan
"""

import os, shutil

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# Path to the directory for orignal dataset and directory for storing for train
data_dir = 'C:/Users/Wei Wan/Documents/DeepLearning/DL with Python/data/train'
base_dir ='C:/Users/Wei Wan/Documents/DeepLearning/DL with Python'

# set directory for train data images
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
# set directory for train cats images
train_cats_dir = os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)
# set directory for train dogs images
train_dogs_dir = os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

# set directory for validation images
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
# set directory for validation cats images
validation_cats_dir = os.path.join(validation_dir,'cats')
os.mkdir(validation_cats_dir)
# set directory for validation dogs images
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)


# set directory for test data images
test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)
# set directory for test cats images
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# set directory for test dogs images
test_dogs_dir = os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)

# Copiesthe first 1000 cat images to train_cat_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(data_dir,fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src,dst)

# Copies the next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(data_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src,dst)
    
# Copies the next 500 cat images to test_cats_dir
fnames=['cat.{}.jpg'.format(i) for i in range (1500,2000)]
for fname in fnames:
    src = os.path.join(data_dir, fname)
    dst=os.path.join(test_cats_dir, fname)
    shutil.copyfile(src,dst)
    
# Copiesthe first 1000 dog images to train_cat_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(data_dir,fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src,dst)

# Copies the next 500 dog images to validation_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(data_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src,dst)
    
# Copies the next 500 dog images to test_cats_dir
fnames=['dog.{}.jpg'.format(i) for i in range (1500,2000)]
for fname in fnames:
    src = os.path.join(data_dir, fname)
    dst=os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src,dst)
    
print('total training cat imagr:' , len(os.listdir(train_cats_dir)))

# listing5.5 installing a small convnet for dogs vs. cats classification
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
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Listing 5.6 Configuring the model for training
model.compile(loss='binary_crossentropy', 
              optimizer = optimizers.RMSprop(lr=1e-4), 
              metrics=['accuracy'])

# Rescales all images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,              # target directory
        target_size=(150,150),  # resizes all images to 150 X 50
        batch_size=20,
        class_mode='binary')    # Because use binary_crossentropy loss,

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')

# Listing5.8 Fitting the model using a batch generator
history =  model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
#history=model.fit_generator(train_generator, steps_per_epoch=100, epochs-30,
#                            validation_data=validation_generator,
#                            validation_steps=50)

# Listing5.9 Saving the model
model.save('cats_and_dogs_small_1.h5')