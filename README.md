# Image Classification Web Service (Flask & FastAPI)

This starter exposes a trained Keras/TensorFlow image classifier as a web API.

## Files
- `app_flask.py` — Flask web service with HTML upload page.
- `app_fastapi.py` — FastAPI service with `/predict` and `/health`.
- `class_names.json` — List of class labels in model prediction index order.
- `requirements.txt` — Python dependencies.
- `templates/index.html` — Simple uploader for Flask.
- `static/style.css` — Minimal styling.
- `predict_client.py` — Example Python client (multipart upload).
- `save_model_snippet.py` — Example code to save a model from a notebook.
- `Dockerfile` — Container to run either Flask or FastAPI.

## Quickstart (Flask)
```bash
pip install -r requirements.txt
# Put your trained model as image_classifier.h5 in this folder (next to app_flask.py)
python app_flask.py
# Open http://127.0.0.1:5000
```

## Quickstart (FastAPI)
```bash
pip install -r requirements.txt
# Put your trained model as image_classifier.h5
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
# Test:
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@sample.jpg"
```

## Expected Model Inputs
- Model file: `image_classifier.h5` (Keras SavedModel/H5) with input shape `(H, W, 3)`.
- Image size: set via env vars `IMG_SIZE=224` or defaults to `224` for transfer learning models.
- Normalization: pixel values are scaled to `[0, 1]`.

## Return Format
```json
{
  "top": [
    {"label": "cat", "proba": 0.91},
    {"label": "dog", "proba": 0.06},
    {"label": "car", "proba": 0.02}
  ],
  "all": {"cat": 0.91, "dog": 0.06, "car": 0.02}
}
```

## Notes
- Update `class_names.json` to match your training label order.
- For CPU-only environments, `tensorflow-cpu` is usually enough.
- If your model expects different preprocessing (e.g., mean/std normalization), modify `preprocess_image()` accordingly.
```