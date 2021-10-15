# deps for data processing
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# explore some data real quick
img = mpimg.imread("./data/train/2b827.png")
print(img.shape)

# get labels in a series
labels = os.listdir("./data/train")
labels = [x[:-4] for x in labels]
labels = pd.Series(labels, name="Labels")

# visual
# plt.imshow(img)
# plt.savefig("img.png")

print(labels.head())

# load in all images to a series
images = os.listdir("./data/train")
images = [mpimg.imread("./data/train/" + x) for x in images]
images = pd.Series(images, name="Images")

print(images.head())

# create a dataframe by concatanating two series
df = pd.concat([images, labels], axis=1)
print(df.head())

# output to a processed csv file for later use
df.to_csv("train.csv")
