from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import io

app = FastAPI()

# Load ViT once at startup
MODEL_ID = "google/vit-base-patch16-224-in21k"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID)
model.eval()  # inference mode

@app.get("/")
def home():
    return {"status": "ViT backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        top5 = torch.topk(probs, k=5)

        labels = [model.config.id2label[idx.item()] for idx in top5.indices[0]]
        scores = [float(s.item()) for s in top5.values[0]]

        return JSONResponse({
            "top5_labels": labels,
            "top5_scores": scores
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
