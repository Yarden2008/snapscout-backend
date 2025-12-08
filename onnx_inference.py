import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("mobilenetv2-10.onnx", providers=["CPUExecutionProvider"])

IMG_SIZE = 224

# Load labels safely
with open("imagenet_classes.txt", "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)

def classify_image(img: Image.Image):
    try:
        input_tensor = preprocess(img)

        inputs = {session.get_inputs()[0].name: input_tensor}

        outputs = session.run(None, inputs)

        if outputs is None or len(outputs) == 0:
            return "Unknown", 0.0

        probs = outputs[0]

        # Validate output shape
        if probs is None or len(probs) == 0:
            return "Unknown", 0.0

        probs = probs[0]

        # If probability vector is wrong length
        if len(probs) != len(LABELS):
            return "Unknown", 0.0

        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])

    except Exception as e:
        print("Model error:", e)
        return "Error", 0.0
