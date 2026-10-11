import sys
import io
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from industrial_fault_detection.config import MODEL_SAVE_PATH, IMAGE_SIZE

# Initialize FastAPI App
app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="API for detecting defects using a Custom ResNet-50 model.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and class names
model = None
class_names = []

@app.on_event("startup")
async def load_model_and_classes():
    """
    Loads the model and class names into memory when the server starts.
    This prevents loading the model on every single API request.
    """
    global model, class_names
    
    if not MODEL_SAVE_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_SAVE_PATH}. Train the model first.")
        
    print("Loading model into memory...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    class_names_path = MODEL_SAVE_PATH.parent / "class_names.txt"
    if class_names_path.exists():
        with open(class_names_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Fallback if text file is missing
        class_names = ["anomaly", "good"]
        
    print(f"Model loaded successfully. Classes: {class_names}")

@app.get("/")
async def root():
    return {"message": "Industrial Anomaly Detection API is live. Send a POST request to /predict."}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, preprocesses it, and runs inference.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        
        image = image.resize(IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        
        
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array, verbose=1)

        prediction_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][prediction_class_idx])
        predicted_class_name = class_names[prediction_class_idx]

        return {
            "filename" : file.filename,
            "prediction" : predicted_class_name,
            "confidence" : round(confidence, 4),
            "raw_probabilities" : predictions[0].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Error processing image: {str(e)}")

    if __name__ == "__main__":
        print("Starting the FastAPI server...")
        uvicorn.run("main:app",host="0.0.0.0",reload=True)