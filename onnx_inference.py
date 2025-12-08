import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession(
    "mobilenetv2-7.onnx",
    providers=["CPUExecutionProvider"]
)

# Load labels
with open("labels.txt", "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0

    # ImageNet normalization (CRITICAL for accuracy)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)
    return arr

def classify_image(img: Image.Image):
    input_tensor = preprocess(img)

    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)[0]  # shape: (1, 1000)
    probs = outputs[0]

    # Get top 5 predictions
    top5 = probs.argsort()[-5:][::-1]
    results = [(LABELS[i], float(probs[i])) for i in top5]

    return results
