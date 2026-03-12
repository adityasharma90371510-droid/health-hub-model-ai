# HealthHub AI

HealthHub AI is an explainable AI system for detecting skin diseases using deep learning and GradCAM visualization.

## Features
- CNN-based skin disease classification
- GradCAM explainability
- Flask web interface
- Drag-and-drop image upload
- AI medical chatbot
- Prediction logging to Excel
- Analytics dashboard (disease distribution, risk distribution, confidence analysis)

## Tech Stack
- Python
- TensorFlow / Keras
- Flask
- OpenCV
- Pandas
- Matplotlib
- Chart.js

## Dataset

This project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset for skin lesion classification.

Dataset Source:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

The dataset contains dermatoscopic images categorized into 7 skin disease classes:

- akiec — Actinic keratoses
- bcc — Basal cell carcinoma
- bkl — Benign keratosis-like lesions
- df — Dermatofibroma
- mel — Melanoma
- nv — Melanocytic nevi
- vasc — Vascular lesions

⚠️ Due to GitHub file size limits, the dataset is **not included in this repository**.

To run the project locally:

1. Download the dataset from Kaggle.
2. Extract it into the project directory:


## Future Work
- Multi-disease diagnosis
- LLM-powered medical assistant
- Cloud database logging
- Prescription and allergy warnings
