import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("mobilenetv3_large.onnx", providers=["CPUExecutionProvider"])

# Load ImageNet labels (1,000 classes)
LABELS = [line.strip() for line in open("labels.txt", "r").readlines()]

def preprocess(img: Image.Image):
    # 1) Resize shortest side to 256
    img = img.convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)))

    # 2) Center crop 224x224
    w, h = img.size
    left = (w - 224) / 2
    top = (h - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # 3) Convert to numpy
    arr = np.array(img).astype("float32") / 255.0

    # 4) Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    # 5) HWC → CHW
    arr = np.transpose(arr, (2, 0, 1))

    # 6) Add batch dimension
    arr = np.expand_dims(arr, axis=0)

    return arr

def classify_image(img: Image.Image):
    x = preprocess(img)
    inp = {session.get_inputs()[0].name: x}

    out = session.run(None, inp)[0][0]

    # Softmax
    exp = np.exp(out - np.max(out))
    probs = exp / exp.sum()

    idx = int(np.argmax(probs))
    label = LABELS[idx]
    score = float(probs[idx])

    return label, score
