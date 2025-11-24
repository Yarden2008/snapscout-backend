import onnxruntime as ort
import numpy as np
from PIL import Image

# ----- Load model -----
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# You may adjust based on your model:
IMG_SIZE = 224
LABELS = [line.strip() for line in open("labels.txt", "r").readlines()]

def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

def classify_image(img: Image.Image):
    input_tensor = preprocess(img)
    inputs = {session.get_inputs()[0].name: input_tensor}

    outputs = session.run(None, inputs)[0]
    probs = outputs[0]

    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])
