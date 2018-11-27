# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:04:42 2018

@author: Wei Wan
"""
import numpy as np
from keras import models
from keras import layers

model = models.Sequential()

def get_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation= 'relu'))
    model.add(layers.Dense(46, activation='softmax'))
    
    return model

num_validation_samples=10000

data = []

# Shuffling the data is usually appropriate. 
np.random.shuffle(data)
# Defines the validation set
validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]

# define training set
training_data=data[:]
test_data = validation_data

# Trains a model on the training data, and evaluates it onthe validation data
model = get_model()
#model.train(training_data)
#validation_scroe=model.evaluation(validation_data)

# At this point you can tune your model,
# etrain it, evaluate it, tune it again...

# onec you've turned your hyperparameters, it's common to train your final 
# model from scratch on all non-test data available. 
model = get_model()
#model.train(np.concatenate([training_data, validation_data]))
#test_score = model.evaluate(test_data)
