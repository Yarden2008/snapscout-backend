from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from onnx_inference import classify_image

app = FastAPI()

@app.get("/")
def home():
    return {"status": "backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        results = classify_image(img)

        return {"predictions": results}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
