from keras_preprocessing.text import Tokenizer
import numpy as np 

samples = ['The cat sat on the mat.','The dog ate my homework.']

# Creates a tokenizer, configured to only take into account 1000 most common words
dimensionality = 1000
max_length = 10
 
results = np.zeros((len(samples),max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split())) [:max_length]:
        # Hashes the word into a random integer index between 0 and 1000
        index = abs(hash(word)) % dimensionality
        results[i,j,index] = 1.

print (results)
