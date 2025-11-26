import cv2
import numpy as np
import pandas as pd
import joblib


# FEATURE EXTRACTOR

def extract_features(path):
    img = cv2.imread(path)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 20])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    area = np.sum(mask > 0)

    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # возвращаем три признака
    return [area, w, h]



# LOAD MODELS

m_height = joblib.load("height_model.pkl")
m_biomass = joblib.load("biomass_model.pkl")
m_leaf = joblib.load("leaf_area_model.pkl")


# RUN PREDICTION
###############-----#
features = extract_features("test.jpg")
###############-----#

df = pd.DataFrame([features], columns=["area_px", "width_px", "height_px"])

print("\n====== Plant Growth Prediction ======")
print(f"🌱 Height:       {m_height.predict(df)[0]:.2f} cm")
print(f"🌿 Biomass:      {m_biomass.predict(df)[0]:.2f} g")
print(f"🍃 Leaf area:    {m_leaf.predict(df)[0]:.2f} cm²")
print("=====================================\n")




# VISUALIZATION
'''def visualize(path):
    img = cv2.imread(path)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 20])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(img_hsv, upper_green, upper_green)

    # Контур
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # Рисуем прямоугольник вокруг растения
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Добавляем подписи
    text = f"Area: {np.sum(mask>0)} px | W:{w} px | H:{h} px"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 150, 255), 2)

    cv2.imshow("Plant Analysis", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

visualize("test2.jpg")
'''