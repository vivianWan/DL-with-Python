# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 22:35:44 2018

@author: Wei Wan
"""

import numpy as np

samples = ['The cat sat on the mat.','The dog ate my homework.']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word in sample.split():
            token_index[word] = len(token_index) +1
            print (token_index)
            
max_length = 10

results = np.zeros(shape = (len(samples), max_length, max(token_index.values()) +1))

for i, sample in enumerate (samples):
    for j, word in list (enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i,j, index] = 1.

print (results)