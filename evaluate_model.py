import numpy as np
import cv2
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model


print("Loading model...")
model = load_model("skin_disease_model.keras", compile=False)
print("Model loaded")


metadata = pd.read_csv("image dataset/HAM10000_metadata.csv")

folder1 = "image dataset/HAM10000_images_part_1"
folder2 = "image dataset/HAM10000_images_part_2"


def get_image_path(image_id):

    path1 = os.path.join(folder1, image_id + ".jpg")
    path2 = os.path.join(folder2, image_id + ".jpg")

    if os.path.exists(path1):
        return path1
    else:
        return path2


metadata["image_path"] = metadata["image_id"].apply(get_image_path)


IMG_SIZE = 96

images = []
labels = []

print("Loading images...")

for _, row in metadata.iterrows():

    img = cv2.imread(row["image_path"])

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    images.append(img)
    labels.append(row["dx"])


X = np.array(images)

print("Running predictions...")
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)


label_map = {
    'akiec':0,
    'bcc':1,
    'bkl':2,
    'df':3,
    'mel':4,
    'nv':5,
    'vasc':6
}

true_classes = [label_map[i] for i in labels]


cm = confusion_matrix(true_classes, predicted_classes)


plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=label_map.keys(),
    yticklabels=label_map.keys()
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Skin Disease Confusion Matrix")

plt.show()


print("\nClassification Report:\n")

print(
classification_report(
true_classes,
predicted_classes,
target_names=label_map.keys()
)
)