# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:55:18 2018

@author: Wei Wan
Classifying movie review
"""

import numpy as np
from keras import models
from keras import layers
from keras import optimizers, losses, metrics
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

model = models.Sequential()
model.add(layers.Dense(16,activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

"""
 training model
 configure the model with the rmsprop optimizer and the binary_crossentropy 
 loss function. Monitor accuracy during training.
"""
# model.compile(optimize = 'rmsprop', loss = 'binary_crossentropy', metrics =['accuracy'])
model.compile(optimizer = optimizers.RMSprop(lr=0.001), 
              loss=  losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
history = model.fit(partial_x_train, partial_y_train, epochs = 20, 
                    batch_size=512, validation_data=(x_val, y_val))
# 
# Plotting the training and validation loss
history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, 21)
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plotting the train and validation accuracy
plt.clf()    # clears the figues
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict ['val_binary_accuracy']

plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validatin accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

predict = model.predict(x_test)

# word_index is a dictionary mapping words to an integer index.
word_index = imdb.get_word_index()
# Reverses it, mapping integer indeices to words
reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])
"""
 Decodes the review. Note that the indices are offset 3 because 0, 1, and 2 are 
 reserved indices for  "padding","start of sequence", and "unknown". 
 """
decoded_review = ' '.join([reverse_word_index.get(i - 3,'?') for i in train_data[0]])
