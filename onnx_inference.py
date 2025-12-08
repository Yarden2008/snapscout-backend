import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession(
    "model/mobilenetv2-10.onnx",
    providers=["CPUExecutionProvider"]
)

# Load labels
with open("labels.txt", "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)
    return arr

def classify_image(img: Image.Image):
    arr = preprocess(img)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})[0]
    
    probs = outputs[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])
