# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:21:02 2018

@author: Wei Wan
"""

import numpy as np
from keras import models
from keras import layers
from keras.datasets import reuters

import matplotlib.pyplot as plt 

# Loading the Retures dataset
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

# Decoding newswires back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])

# the indices are offset by 3 because 0, 1, and 2 are reserved indices for 
# "padding", "start of sequence", and "unknow"
decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

# Encoding the data
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# Mondel definition
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation= 'relu'))
model.add(layers.Dense(46, activation='softmax'))

# Vectorized training data and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Vectorized train labels and test labels
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
# building way to do in Keras:
# one_hot_train_labels = to_categorical(train_labels)

# Complling the model
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics= ['accuracy'])

# Setting aside a validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# train the network for 20 epochs
history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size = 512, validation_data=(x_val,y_val))

history_dict = history.history
history_dict.keys()

# Plotting the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plotting the training and validation accuracy
plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# retraining a model from scratch
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation= 'relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val,y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# Generating prediction for new data
predictions = model.predict(x_test)

# Each entry in predictioans is a vector of length 46:
print (predictions[0].shape)

# The coefficients in this vector sume to 1:
print (np.sum(predictions[0])) 

# The largest entry is the predicated class - the clas with the highest probability
print (np.argmax(predictions[0]))

