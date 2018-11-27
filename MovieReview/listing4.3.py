# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:37:45 2018
DL with Python 
overfitting and underfitting
Listing 4.3 -- Listing 4.6
@author: Wei Wan
"""

import numpy as np

from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import regularizers
from keras.datasets import imdb


import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    # Creates an all-zeros matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences):
        # Sets specific indixes of result[i] to 1s
        results[i,sequence] = 1.
    
    return results

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)


# vectorized training data and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# origianl model
model = models.Sequential()
model.add(layers.Dense(16,activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# listing 4.4 Version of the model with lower capacity
model1 = models.Sequential()
model1.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model1.add(layers.Dense(4, activation='relu'))
model1.add(layers.Dense(1, activation='sigmoid'))

# listing 4.5 Version of the model with lower capacity
model2 = models.Sequential()
model2.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))

# setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.compile(optimizer = optimizers.RMSprop(lr=0.001), 
              loss=  losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
history = model.fit(partial_x_train, partial_y_train, epochs = 20, 
                    batch_size=512, validation_data=(x_val, y_val))
# 
# Plotting the training and validation loss
history_dict = history.history
loss_values = history_dict['val_loss']

# smaller model
model1.compile(optimizer = optimizers.RMSprop(lr=0.001), 
              loss=  losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
history1 = model.fit(partial_x_train, partial_y_train, epochs = 20, 
                    batch_size=512, validation_data=(x_val, y_val))
# Plotting the training and validation loss
history_dict1 = history1.history
loss_values1 = history_dict1['val_loss']

# Bigger model
model2.compile(optimizer = optimizers.RMSprop(lr=0.001), 
              loss=  losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
history2 = model.fit(partial_x_train, partial_y_train, epochs = 20, 
                    batch_size=512, validation_data=(x_val, y_val))
 
# Plotting the training and validation loss
history_dict2 = history2.history
loss_values2 = history_dict2['val_loss']

epochs = range(1, 21)
plt.plot(epochs, loss_values, 'bo', label = 'Orifinal One')
plt.plot(epochs, loss_values1, 'go', label = 'Smaller Model')
plt.plot(epochs, loss_values2, 'ro', label = 'Bigger Model')
plt.title('Validation loss')
plt.xlabel('Epochs')
plt.xlim((0,20))
plt.ylabel('Validation Loss')
plt.legend()

plt.show()

# model adding L2 weigh regularization (list4.6)
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                       activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                       activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Listing 4.8 Ading dropout to the IMDB network
model = models.Sequential()
model.add(layers.Dense(6, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16,activaton='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))