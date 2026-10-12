import sys
import io
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Dynamically resolve the project root and add the 'src' directory to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from industrial_fault_detection.config import MODEL_SAVE_PATH, IMAGE_SIZE

# Global variables for model and class names
model = None
class_names = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager replaces the deprecated @app.on_event("startup").
    Loads the model into memory before the server starts accepting requests.
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
    
    # Yield control back to FastAPI to handle requests
    yield
    
    # Cleanup on shutdown
    print("Shutting down API and clearing memory...")
    model = None
    class_names = []

# Initialize FastAPI App with the lifespan manager
app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="API for detecting defects using a Custom ResNet-50 model.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS so frontend dashboards can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # Read the image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess the image to match the ResNet-50 input
        image = image.resize(IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        
        # Add the batch dimension (shape becomes: 1, 224, 224, 3)
        img_array = tf.expand_dims(img_array, 0)
        
        # Run inference
        predictions = model.predict(img_array, verbose=0)
        
        # Extract results
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class_name = class_names[predicted_class_idx]
        
        return {
            "filename": file.filename,
            "prediction": predicted_class_name,
            "confidence": round(confidence, 4),
            "raw_probabilities": predictions[0].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)