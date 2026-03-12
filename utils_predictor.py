import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 96

classes = [
"akiec",
"bcc",
"bkl",
"df",
"mel",
"nv",
"vasc"
]

model = load_model("skin_disease_model.keras", compile=False)


def predict_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

    img = img / 255.0

    img = np.expand_dims(img,axis=0)

    preds = model.predict(img, verbose=0)

    class_index = np.argmax(preds)

    label = classes[class_index]

    confidence = float(preds[0][class_index])

    probabilities = preds[0]

    return label, confidence, probabilities