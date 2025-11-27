import onnxruntime as ort
import numpy as np
from PIL import Image

# ------------------------------
# Load ONNX model
# ------------------------------
session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]
)

# Print input/output details once at server startup
inp = session.get_inputs()[0]
out = session.get_outputs()[0]
print("INPUT NAME:", inp.name)
print("INPUT SHAPE:", inp.shape)
print("INPUT TYPE:", inp.type)
print("OUTPUT NAME:", out.name)
print("OUTPUT SHAPE:", out.shape)
print("OUTPUT TYPE:", out.type)

# ------------------------------
# ImageNet mean/std for MobileNetV3
# ------------------------------
IMG_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Load labels
LABELS = [line.strip() for line in open("labels.txt", "r").readlines()]


# ------------------------------
# Preprocessing
# ------------------------------
def preprocess(img: Image.Image):
    # Resize to 224x224
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy (HWC)
    arr = np.array(img).astype("float32") / 255.0

    # Normalize
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # Convert to (C, H, W)
    arr = np.transpose(arr, (2, 0, 1))

    # Add batch dimension → (1, C, H, W)
    arr = np.expand_dims(arr, axis=0)

    return arr.astype("float32")


# ------------------------------
# Inference
# ------------------------------
def classify_image(img: Image.Image):
    input_tensor = preprocess(img)

    inputs = {session.get_inputs()[0].name: input_tensor}

    # Run model
    outputs = session.run(None, inputs)[0]

    # Softmax
    probs = np.squeeze(outputs)
    exp = np.exp(probs - np.max(probs))
    softmax = exp / np.sum(exp)

    # Top-1
    idx = int(np.argmax(softmax))
    confidence = float(softmax[idx])

    label = LABELS[idx]

    return label, confidence
