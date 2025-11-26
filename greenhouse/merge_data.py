import pandas as pd

features = pd.read_csv("features.csv")
labels = pd.read_csv("dataset_fixed.csv")

merged = features.merge(labels, on="filename")
merged.to_csv("training_data.csv", index=False)

print("training_data.csv готов!")
