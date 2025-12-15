import json
import onnxruntime as ort
import numpy as np
from PIL import Image

# ---------- LOAD LABELS ----------
with open("labels.json", "r") as f:
    raw = json.load(f)

# מקרה 1: "0": "tench"
if isinstance(raw["0"], str):
    LABELS = [raw[str(i)] for i in range(1000)]

# מקרה 2: "0": ["n01440764", "tench"]
else:
    LABELS = [raw[str(i)][1] for i in range(1000)]

assert len(LABELS) == 1000, "Labels must be exactly 1000"


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

session = ort.InferenceSession(
    "mobilenetv2-10.onnx",
    providers=["CPUExecutionProvider"]
)

IMG_SIZE = 224

def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - 0.5) / 0.5   # normalization
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr
def classify_image(img: Image.Image):
    x = preprocess(img)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: x})
    probs = outputs[0][0]

    idx = int(np.argmax(probs))

    if idx >= len(LABELS):
        return "unknown", float(probs[idx])

    return LABELS[idx], float(probs[idx])
