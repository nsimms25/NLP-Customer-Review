import pandas as pd

print(pd.__version__)

data = pd.read_csv("data/raw/Reviews.csv")

print(data.columns)