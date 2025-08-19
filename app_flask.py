import os, io, json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

APP_TITLE = "Image Classifier (Flask)"
MODEL_PATH = os.getenv("MODEL_PATH", "image_classifier.h5")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES", "class_names.json")

app = Flask(__name__)

# Lazy-load model to reduce cold start time if needed
model = None
class_names = None

def load_model():
    global model, class_names
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place image_classifier.h5 next to app_flask.py.")
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
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def postprocess_probs(probs: np.ndarray):
    probs = probs.flatten()
    top_idx = probs.argsort()[::-1][:3]
    top = [{"label": class_names[i], "proba": float(probs[i])} for i in top_idx]
    all_map = {class_names[i]: float(p) for i, p in enumerate(probs)}
    return {"top": top, "all": all_map}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", title=APP_TITLE)

@app.route("/health", methods=["GET"])
def health():
    try:
        load_model()
        return jsonify({"status": "ok", "model_loaded": True, "img_size": IMG_SIZE})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model()
        if "file" not in request.files:
            return jsonify({"error": "No file part 'file' in form-data"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        arr = preprocess_image(file.read())
        probs = model.predict(arr, verbose=0)
        result = postprocess_probs(probs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
