from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

app = FastAPI()

# Allow your Android app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (Vision Transformer)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

@app.get("/")
def root():
    return {"status": "up", "message": "SnapScout AI model is running"}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(file.file).convert("RGB")

        # Preprocess
        inputs = processor(images=image, return_tensors="pt")

        # Run model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get top 5 predictions
        logits = outputs.logits
        softmax = torch.nn.functional.softmax(logits, dim=1)
        top5 = torch.topk(softmax, k=5)

        labels = [model.config.id2label[idx.item()] for idx in top5.indices[0]]
        scores = [round(score.item(), 4) for score in top5.values[0]]

        return {
            "labels": labels,
            "scores": scores
        }

    except Exception as e:
        return {"error": str(e)}
