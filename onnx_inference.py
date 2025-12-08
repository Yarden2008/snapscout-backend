import os
import urllib.request
import onnxruntime as ort
import numpy as np
from PIL import Image

MODEL_URL = "https://drive.google.com/uc?export=download&id=1YY_9IR8fY-jjWlZs1bz0Ovi38upaZeoN"
MODEL_PATH = "efficientnet-lite4.onnx"
LABELS_PATH = "imagenet_classes.txt"

# EfficientNet Lite uses 224x224 input
IMG_SIZE = 224


# ---------------------------
# AUTO DOWNLOAD UTIL
# ---------------------------
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("➡ Downloading model from Google Drive...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✔ Model downloaded successfully")

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"Missing {LABELS_PATH}. Please upload it to the repo."
        )


# ---------------------------
# MODEL LOADING
# ---------------------------
download_model_if_needed()

print("➡ Loading ONNX model...")
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
print("✔ Model ready")

LABELS = [line.strip() for line in open(LABELS_PATH, "r").readlines()]


# ---------------------------
# PREPROCESSING (EfficientNet)
# ---------------------------
def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0

    # EfficientNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    # HWC → CHW
    arr = np.transpose(arr, (2, 0, 1))

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)

    return arr.astype("float32")


# ---------------------------
# INFERENCE
# ---------------------------
def classify_image(img: Image.Image):
    input_tensor = preprocess(img)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    logits = outputs[0][0]
    idx = int(np.argmax(logits))
    confidence = float(np.max(softmax(logits)))

    return LABELS[idx], confidence


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)
