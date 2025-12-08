from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from onnx_inference import classify_image
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def home():
    return {"status": "mobilenetv2-10 backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        label, score = classify_image(image)

        return {
            "label": label,
            "confidence": round(score, 4)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
