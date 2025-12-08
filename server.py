from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from onnx_inference import classify_image

app = FastAPI()

@app.get("/")
def home():
    return {"status": "EfficientNet-Lite4 backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        label, confidence = classify_image(image)

        return {
            "label": label,
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
