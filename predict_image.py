import numpy as np
import cv2
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("skin_disease_model.keras", compile=False)
print("Model loaded successfully")


classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

IMG_SIZE = 96


def predict_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Image not found!")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print("\nPrediction:", classes[predicted_class])
    print("Confidence:", float(confidence))


# CHANGE IMAGE PATH HERE
image_path = "image dataset/HAM10000_images_part_1/ISIC_0024306.jpg"

predict_image(image_path)