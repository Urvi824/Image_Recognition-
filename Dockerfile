# Build a minimal image for CPU inference
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Default to Flask (change to FastAPI below if preferred)
ENV MODEL_PATH=/app/image_classifier.h5
ENV CLASS_NAMES=/app/class_names.json
ENV IMG_SIZE=224
ENV PORT=5000

# Expose Flask port
EXPOSE 5000

# --- Flask ---
CMD ["python", "app_flask.py"]

# --- For FastAPI, use instead:
# EXPOSE 8000
# CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
