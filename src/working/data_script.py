import os
import pandas as pd

# create the annotations file
root_dir = "data/"
df = pd.DataFrame()

# create cols
df["Paths"] = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
df["Labels"] = [x[:-4] for x in os.listdir(root_dir)]

# save the file
df.to_csv("data.csv")
