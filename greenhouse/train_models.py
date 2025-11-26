import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("training_data.csv")

X = df[["area_px", "width_px", "height_px"]]

# HEIGHT
y_height = df["height"]
model_height = LinearRegression().fit(X, y_height)
joblib.dump(model_height, "height_model.pkl")

# BIOMASS
y_biomass = df["biomass"]
model_biomass = LinearRegression().fit(X, y_biomass)
joblib.dump(model_biomass, "biomass_model.pkl")

# LEAF AREA
y_leaf = df["leaf_area"]
model_leaf = LinearRegression().fit(X, y_leaf)
joblib.dump(model_leaf, "leaf_area_model.pkl")

print("Все модели сохранены!")
