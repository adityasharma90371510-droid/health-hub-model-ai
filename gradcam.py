import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

IMG_SIZE = 96

model = load_model("skin_disease_model.keras", compile=False)

last_conv_layer_name = "Conv_1"


def generate_gradcam(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    input_image = np.expand_dims(img, axis=0)

    preds = model.predict(input_image)
    class_index = np.argmax(preds[0])

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(input_image)

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = heatmap.numpy() if hasattr(heatmap, "numpy") else heatmap

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(image_path)
    original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    output_filename = "gradcam.jpg"
    output_path = os.path.join("static/uploads", output_filename)

    cv2.imwrite(output_path, superimposed)

    return output_filename