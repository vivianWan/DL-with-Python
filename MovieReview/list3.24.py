# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:56:36 2018

@author: Wei Wan
"""
import numpy as np
from keras import layers
from keras import models
from keras.datasets import boston_housing

import matplotlib.pyplot as plt

def build_model():
    # because we will need to instantiate the same model multiple times, 
    # we use a function to construct it.
    model= models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile (optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    return model
    
(train_data, train_targets),(test_data, test_targets) = boston_housing.load_data()

# Normallzing the data
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std

test_data -= mean
test_data /= std

#K-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print ('processing fold :', i)
    # prepares the validation data: data from artition #k:
    val_data= train_data[i*num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i*num_val_samples: (i+1)*num_val_samples]
    
    # Perpare the training data: data from all other partition
    partial_train_data = np.concatenate(
            [train_data[:i* num_val_samples],
             train_data[(i+1)*num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate(
            [train_targets[:i*num_val_samples],
             train_targets[(i+1)*num_val_samples:]], axis = 0)
    
    # builds the Keras odel (already compiled)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, 
                        epochs = num_epochs, batch_size =1, verbose = 0)
    print (history.history.keys())
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
           


# Saving the validation logs at each fold
num_epochs = 500
all_mae_histories =[]
for i in range(k):
    print ('processing fold Num: ', i)
    val_data = train_data[i*num_val_samples:(i+1)* num_val_samples]
    val_targets =train_targets[i*num_val_samples:(i+1)*num_val_samples]
   
    partial_train_data = np.concatenate(
            [train_data[:i* num_val_samples],
            train_data[(i+1)*num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate(
            [train_targets[:i*num_val_samples],
             train_targets[(i+1)*num_val_samples:]], axis = 0)
    
    # builds the Keras odel (already compiled)
    model = build_model()    
    history = model.fit(partial_train_data, partial_train_targets, 
                        validation_data=(val_data, val_targets),
              epochs = num_epochs, batch_size =1, verbose = 0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
# building the history of successive mean K-fold validation scores
average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# Plotting validation scores
plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.ylim((2.0,4.5))
plt.show()

## plotting validation scores, excluding the firsst 1- data points
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1] *factor
            current_point = previous + point *(1-factor)
            smoothed_points.append(current_point)
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1,len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.ylim((2.2,2.9))
plt.xlim((0,500))
plt.show()

           
           