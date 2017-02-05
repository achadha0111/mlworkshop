import pandas as pd
import numpy as np
import gensim.models.word2vec as w2v
import codecs
import glob
import multiprocessing
import os
import re
import pprint
import sklearn.manifold
import matplotlib.pyplot as plt

print (os.getcwd())

songs = pd.read_csv("/Users/aayushchadha/Downloads/songdata.csv", header=0)
#songs.head()
songs = songs[songs.artist != 'Lata Mangeshkar']
songs.head()

songs2vec = w2v.Word2Vec.load("trained/songs2vec.w2v")

import string

def songVector(row):
    vector_sum = 0
    words = row.lower().split()
    for word in words:
        vector_sum = vector_sum + songs2vec[word]
    vector_sum = vector_sum.reshape(1,-1)
    normalised_vector_sum = sklearn.preprocessing.normalize(vector_sum)
    return normalised_vector_sum

from sklearn.model_selection import train_test_split

train, test = train_test_split(songs, test_size = 0.9)



train['song_vector'] = train['text'].apply(songVector)


train.head()
