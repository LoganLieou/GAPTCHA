# deps for data processing
import os
import pandas as pd

# init our dataframe
df = pd.DataFrame()

# define the root dir that contains data
root = "data/train"

# load in all images to a series
images = [os.path.join(root, x) for x in os.listdir(root)]
df["Images"] = images

# get labels in a series
labels = [x[:-4] for x in os.listdir(root)]
df["Labels"] = labels

# output to a processed csv file for later use
df.to_csv("data.csv")
