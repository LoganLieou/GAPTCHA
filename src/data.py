import os
import string
import numpy as np
from matplotlib.image import imread

ims = [imread("data/train/" + x) for x in os.listdir("data/train/")]
lbs = [x[:-4] for x in os.listdir("data/train/")]

symbols = string.printable

def forward_map(word):
    one_hot_word = np.zeros((len(word), len(symbols)))
    for n, letter in enumerate(word):
        one_hot_word[n][symbols.index(letter)] = 1
    return one_hot_word

def inverse_map(one_hot_word):
    word = ""
    for i in range(len(one_hot_word)):
        for j in range(len(one_hot_word[0])):
            if one_hot_word[i][j] >= 1.:
                word += symbols[j]
    return word

y = np.array([forward_map(x) for x in lbs])
X = np.array([x[:, :, 0] for x in ims])
X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))

print(y.shape)
print(X.shape)
