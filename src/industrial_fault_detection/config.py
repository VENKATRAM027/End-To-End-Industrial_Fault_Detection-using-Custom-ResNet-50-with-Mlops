import yaml
from pathlib import Path

# Resolves to the root folder (4 levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load parameters from params.yaml
PARAMS_PATH = BASE_DIR / "params.yaml"
with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

# Extract parameters
CATEGORY = params["base"]["category"]
RANDOM_SEED = params["base"]["random_seed"]

IMAGE_SIZE = tuple(params["train"]["image_size"])
BATCH_SIZE = params["train"]["batch_size"]
EPOCHS = params["train"]["epochs"]
LEARNING_RATE = float(params["train"]["learning_rate"])

# Dynamic Naming
MODEL_NAME = f"Custom_ResNet50_{CATEGORY}"

# Define main directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR  # Mapped for evaluate.py compatibility

# Data paths
TRAIN_DIR = PROCESSED_DATA_DIR / CATEGORY / "train"
VAL_DIR = PROCESSED_DATA_DIR / CATEGORY / "val"

# Outputs
MODEL_SAVE_PATH = MODELS_DIR / f"{CATEGORY}_best_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.txt"