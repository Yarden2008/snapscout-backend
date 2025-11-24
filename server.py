from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from onnx_inference import classify_image
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ONNX backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        # Run ONNX model
        label, score = classify_image(image)

        return JSONResponse({
            "label": label,
            "confidence": float(score)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
