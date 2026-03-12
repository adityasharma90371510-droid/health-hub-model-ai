import pandas as pd

# Load metadata
metadata = pd.read_csv("image dataset/HAM10000_metadata.csv")

print("Dataset shape:")
print(metadata.shape)

print("\nFirst 5 rows:")
print(metadata.head())

print("\nDisease classes:")
print(metadata["dx"].value_counts())
import os

# image folders
image_folder1 = "image dataset/HAM10000_images_part_1"
image_folder2 = "image dataset/HAM10000_images_part_2"

# function to get image path
def get_image_path(image_id):
    
    path1 = os.path.join(image_folder1, image_id + ".jpg")
    path2 = os.path.join(image_folder2, image_id + ".jpg")
    
    if os.path.exists(path1):
        return path1
    else:
        return path2

# create new column
metadata["image_path"] = metadata["image_id"].apply(get_image_path)

print("\nImage paths added:")
print(metadata[["image_id","image_path"]].head())
from sklearn.preprocessing import LabelEncoder

# Encode disease labels
encoder = LabelEncoder()
metadata["label"] = encoder.fit_transform(metadata["dx"])

print("\nEncoded labels:")
print(metadata[["dx","label"]].head())
import cv2
import numpy as np

IMG_SIZE = 64

images = []
labels = []

for index, row in metadata.iterrows():

    img = cv2.imread(row["image_path"])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    images.append(img)
    labels.append(row["label"])

X = np.array(images)
y = np.array(labels)

print("\nImage dataset shape:")
print(X.shape)
# Normalize pixel values
X = X / 255.0

print("\nSample pixel values after normalization:")
print(X[0][0][0])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

print("\nExample encoded label:")
print(y_train[0])

print(metadata[["dx","label"]].drop_duplicates())