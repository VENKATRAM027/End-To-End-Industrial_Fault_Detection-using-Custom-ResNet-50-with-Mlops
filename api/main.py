
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

from industrial_fault_detection.config import (
    MODEL_SAVE_PATH,
    MODELS_DIR,
    IMAGE_SIZE,
)


model = None
class_names = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names

    if not MODEL_SAVE_PATH.exists():
        raise RuntimeError(
            f"Model not found: {MODEL_SAVE_PATH}. Train the model first."
        )

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    names_path = MODELS_DIR / "class_names.json"
    if names_path.exists():
        class_names = json.loads(names_path.read_text(encoding="utf-8"))
    else:
        # Compatibility with a model trained by the earlier pipeline.
        legacy_path = MODELS_DIR / "class_names.txt"
        if not legacy_path.exists():
            raise RuntimeError("Model class names file is missing.")
        class_names = [
            line.strip()
            for line in legacy_path.read_text(encoding="utf-8").splitlines()
        ]

    if len(class_names) != int(model.output_shape[-1]):
        raise RuntimeError("Class names do not match the model output.")

    yield

    model = None
    class_names = []


app = FastAPI(
    title="Industrial Fault Detection API",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Industrial Fault Detection API", "docs": "/docs"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": model.name}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        with Image.open(io.BytesIO(contents)) as image:
            image = image.convert("RGB")
            image = image.resize(
                (IMAGE_SIZE[1], IMAGE_SIZE[0]),
                Image.Resampling.BILINEAR,
            )
            array = np.asarray(image, dtype=np.float32)

        # Must match the training pipeline's preprocessing.
        array = tf.keras.applications.resnet50.preprocess_input(array)
        batch = np.expand_dims(array, axis=0)

        probabilities = model.predict(batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))

        return {
            "filename": file.filename,
            "prediction": class_names[predicted_index],
            "confidence": round(float(probabilities[predicted_index]), 4),
            "probabilities": {
                name: round(float(probabilities[i]), 4)
                for i, name in enumerate(class_names)
            },
        }

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed.")
