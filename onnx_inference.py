import onnxruntime as ort
import numpy as np
import json
from PIL import Image

# ------------ LOAD MODEL ------------
MODEL_PATH = "mobilenetv2-10.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Print model info (useful for debugging)
print("INPUT:", session.get_inputs()[0].name, session.get_inputs()[0].shape)
print("OUTPUT:", session.get_outputs()[0].name, session.get_outputs()[0].shape)

# ------------ LOAD LABELS ------------
IMAGENET_CLASSES = [c.strip() for c in open("imagenet_classes.txt", "r").readlines()]
SNAPSCOUT_LABELS = [c.strip() for c in open("labels.txt", "r").readlines()]

# ------------ LOAD CATEGORY MAP ------------
with open("category_map.json", "r") as f:
    CATEGORY_MAP = json.load(f)

IMG_SIZE = 224


def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)
    return arr


def imagenet_to_snapscout(class_name: str):
    """
    Converts raw ImageNet class → SnapScout category
    using semantic keyword matching.
    """
    class_name = class_name.lower()

    for snap_cat, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in class_name:
                return snap_cat

    # If nothing matched → return unknown
    return "unknown"


def classify_image(img: Image.Image):
    input_tensor = preprocess(img)
    inputs = {session.get_inputs()[0].name: input_tensor}

    # Run inference
    outputs = session.run(None, inputs)[0]
    probs = outputs[0]

    # Top-5 indices
    top5_idx = np.argsort(probs)[-5:][::-1]

    # Map them to SnapScout categories
    mapped = []
    for idx in top5_idx:
        imagenet_label = IMAGENET_CLASSES[idx]
        score = float(probs[idx])
        snap_cat = imagenet_to_snapscout(imagenet_label)
        mapped.append({"snap_category": snap_cat, "imagenet": imagenet_label, "confidence": score})

    # Sort by confidence but prioritize known categories
    mapped.sort(key=lambda x: (-x["confidence"], x["snap_category"] != "unknown"))

    # Return top 1 SnapScout category
    best = mapped[0]
    return best["snap_category"], best["confidence"]
