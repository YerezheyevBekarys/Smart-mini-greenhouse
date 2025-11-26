import cv2
import numpy as np
import pandas as pd
import os

def extract_features(image_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Зелёный диапазон
    lower_green = np.array([25, 40, 20])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # Площадь зелёных пикселей
    green_area = np.sum(mask > 0)

    # Bounding box
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    return green_area, w, h

# Генерация таблицы
images_dir = "images"
rows = []

for filename in os.listdir(images_dir):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(images_dir, filename)
        area, w, h = extract_features(path)
        rows.append({
            "filename": filename,
            "area_px": area,
            "width_px": w,
            "height_px": h
        })

df = pd.DataFrame(rows)
df.to_csv("features.csv", index=False)
print("Saved features.csv")
