import onnxruntime as ort
import numpy as np
from PIL import Image
import json

# ----- Load model -----
session = ort.InferenceSession("mobilenetv2-7.onnx", providers=["CPUExecutionProvider"])

# Load 1000 ImageNet class names
with open("imagenet-simple-labels.json", "r") as f:
    LABELS = json.load(f)

IMG_SIZE = 224

def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

def classify_image(img: Image.Image):
    input_tensor = preprocess(img)
    inputs = {session.get_inputs()[0].name: input_tensor}

    outputs = session.run(None, inputs)[0]   # shape: (1, 1000)
    probs = outputs[0]

    idx = int(np.argmax(probs))
    label = LABELS[idx] if 0 <= idx < len(LABELS) else f"class_{idx}"
    return label, float(probs[idx])

