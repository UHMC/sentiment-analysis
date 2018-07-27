# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:16:23 2018

@author: Saxon Knight
"""

# imports
from re import sub
from collections import Counter
import tensorflow as tf
from tensorflow import keras

# settings
max_vocab_size = 5000
num_embedding_dims = 16
num_hidden_nodes = 16

# list of files of lines of tab-separated values
text_dataset_filenames = [
        'sentiment labelled sentences\\amazon_cells_labelled.txt',
        'sentiment labelled sentences\\imdb_labelled.txt',
        'sentiment labelled sentences\\yelp_labelled.txt'
        ]

# extract data and labels from files
data = []
labels = []
for filename in text_dataset_filenames:
    with open(filename) as file:
        for line in file:
            data.append(line[:-3])
            labels.append(int(line[-2]))

# clean up data
for i in range(len(data)):
    data[i] = sub('[^a-z0-9 ]+', '', data[i].lower())

# get sorted (most frequent first) vocabulary
all_words = []
for line in data:
    for word in line.split():
        all_words.append(word)
word_frequency = Counter(all_words)
vocab_tuples_sorted = list(word_frequency.most_common())
vocab = [vocab_tuple[0] for vocab_tuple in vocab_tuples_sorted]
vocab.append('unknownword')

# make numeric ID encoder
def encode_text_to_IDs(text):
    words = text.split()
    ids = []
    for word in words:
        try:
            ids.append(vocab.index(word))
        except IndexError:
            ids.append(-1)
    return ids

# make numeric ID decoder
def decode_IDs_to_text(IDs):
    return ' '.join([vocab[ID] for ID in IDs])

# make model
model = keras.Sequential()
model.add(keras.layers.Embedding(max_vocab_size, num_embedding_dims))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(num_hidden_nodes, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# train model


# test model