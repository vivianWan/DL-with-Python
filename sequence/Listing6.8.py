# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:30:31 2018
Deep Learning with Python
Listing 6.8 Processing the labels of the raw IMDB data

@author: Wei Wan
"""
import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# set the directories
imdb_dir = 'C:/Users/Wei Wan/Documents/DeepLearning/DL with Python/sequence/aclImdb'
train_dir = os.path.join(imdb_dir,'test')

labels = []
texts =[]

for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.list(dir_name):
        f = open(os.path.join(dir_name, fname))
        texts.append(f.read())
        f.close()
        if label_type == 'neg':
            labels.append(0)
        else:
            labels.append(1)
            
# Tokenizing the data Listing 6.9 Tokenizing the text of the raw IMDB data
# Cuts off reviews after 100 words
maxlen = 100
# Trains on 200 samples
training_samples = 200
# validates on 10000 samples
validation_samples = 10000
# Considers only the top 10000 words in the dataset
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index=tokenizer.word_index
print ('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print ('Shape of data tensor:', data.shape)
print ('Shape of label tensor:', labels.shape)

# splits the data inot a training set and a validation set
# but first shuffles the data, because you're starting with tdata in which samples
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = 
