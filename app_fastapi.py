import os, io, json
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf

APP_TITLE = "Image Classifier (FastAPI)"
MODEL_PATH = os.getenv("MODEL_PATH", "image_classifier.h5")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES", "class_names.json")

app = FastAPI(title=APP_TITLE)

model = None
class_names = None

def load_model():
    global model, class_names
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place image_classifier.h5 next to app_fastapi.py.")
        model = tf.keras.models.load_model(MODEL_PATH)
    if class_names is None:
        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(f"class_names.json not found at {CLASS_NAMES_PATH}.")
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)

def preprocess_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def postprocess_probs(probs: np.ndarray):
    probs = probs.flatten()
    top_idx = probs.argsort()[::-1][:3]
    top = [{"label": class_names[i], "proba": float(probs[i])} for i in top_idx]
    all_map = {class_names[i]: float(p) for i, p in enumerate(probs)}
    return {"top": top, "all": all_map}

@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model_loaded": True, "img_size": IMG_SIZE}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        load_model()
        content = await file.read()
        arr = preprocess_image(content)
        probs = model.predict(arr, verbose=0)
        result = postprocess_probs(probs)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
