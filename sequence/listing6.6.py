import numpy as np 
import tensorflow as tf 
from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense

"""
The Embedding layer takes at least two arguments: 
    the number of possible tokens (here, 100:1 + maximum word index) and 
    the dimensionality of the embeddings (here, 64)
"""
embedding_layer = Embedding(1000, 64)

"""
Loading the IMDB data fro use with an Embedding layer
"""
#number of words to consdier as features
max_features = 10000

# Cuts off the text after this number of words (among the max_features most common words)
max_len = 20

# Loads the data as lists of integers
(x_train, y_train),(x_test,y_test) = imdb.load_data(num_words= max_features)

# Turns the lists of integers into a 2D integer tensor of shape (samples, max_len)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen= max_len)

"""
Using an Embedding layer and classifier on the IMDB data
"""
model = Sequential()
# Specifies the maximum input length to the Embedding layer so that you can later flatten 
# the embeeded inputs. After the Embedding layer, the activations have shape (samples, max_len,8).
model.add(Embedding(10000, 8, input_length=max_len))

# Flatten the 3D tensor of embeddings into a 2D tensor of shape (samples, max_len*8)
model.add(Flatten())

# Adds the classifier on top
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics =['acc'])
model.summary()

history = model.fit(x_train,y_train, epochs= 10, batch_size= 32, validation_split=0.2)
