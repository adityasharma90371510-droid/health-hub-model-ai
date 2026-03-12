import pandas as pd
import os
import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# =============================
# Load Dataset Metadata
# =============================

metadata = pd.read_csv("image dataset/HAM10000_metadata.csv")

image_folder1 = "image dataset/HAM10000_images_part_1"
image_folder2 = "image dataset/HAM10000_images_part_2"


def get_image_path(image_id):

    path1 = os.path.join(image_folder1, image_id + ".jpg")
    path2 = os.path.join(image_folder2, image_id + ".jpg")

    if os.path.exists(path1):
        return path1
    else:
        return path2


metadata["image_path"] = metadata["image_id"].apply(get_image_path)


# =============================
# Encode Labels
# =============================

encoder = LabelEncoder()
metadata["label"] = encoder.fit_transform(metadata["dx"])


# =============================
# Load Images
# =============================

IMG_SIZE = 96

images = []
labels = []

for index, row in metadata.iterrows():

    img = cv2.imread(row["image_path"])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    images.append(img)
    labels.append(row["label"])


X = np.array(images)
y = np.array(labels)

X = X / 255.0


# =============================
# Train Test Split
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)


# =============================
# Data Augmentation
# =============================

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)


# =============================
# MobileNetV2 Base Model
# =============================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)


# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Fine tune last layers
for layer in base_model.layers[-30:]:
    layer.trainable = True


# =============================
# Custom Classification Head
# =============================

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


# =============================
# Compile Model
# =============================

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =============================
# Train Model
# =============================

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=15,
    validation_data=(X_test, y_test)
)


# =============================
# Save Model
# =============================

model.save("skin_disease_model.keras")


# =============================
# Prediction Test
# =============================

classes = ["akiec","bcc","bkl","df","mel","nv","vasc"]

def predict_image(path):

    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img / 255.0

    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)

    idx = np.argmax(pred)
    conf = pred[0][idx]

    print("\nPrediction:",classes[idx])
    print("Confidence:",conf)


predict_image("image dataset/HAM10000_images_part_1/ISIC_0027419.jpg")


# =============================
# Accuracy Graph
# =============================

plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# =============================
# Loss Graph
# =============================

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()