import os
import string
import numpy as np
from matplotlib.image import imread

ims = [imread("data/train/" + x) for x in os.listdir("data/train/")]
lbs = [x[:-4] for x in os.listdir("data/train/")]

symbols = string.printable

Y = np.zeros((len(lbs), len(symbols)))

# mapping stolen from someone else :)
for n, name in enumerate(lbs):
    for ln, letter in enumerate(name):
        try:
            Y[n][symbols.index(letter)] = 1
        except:
            print(letter, end=" ")

# big brain mapping
X = []
for i in range(len(ims)):
    X.append(ims[i][:, :, 0])
X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
