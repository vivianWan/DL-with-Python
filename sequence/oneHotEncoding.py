from keras_preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.','The dog ate my homework.']

# Creates a tokenizer, configured to only take into account 1000 most common words
tokenizer = Tokenizer(num_words=1000)
# Builds the word index
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)   # Turns strings into lists of integer indices

# You could also directly get the one-hot binary representations. 
# Vectorization modes other than one-hot encoding are supported by this tokenizer 
one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')

# How you can recover the word index that was computed
word_index = tokenizer.word_index

print ('Found %s unique tokens.' % len(word_index))
print (sequences)