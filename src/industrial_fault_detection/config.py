from pathlib import Path

# DIRECTORY SETUP
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Define main directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# PROJECT & DATA SETTINGS
CATEGORY = "bottle"
MODEL_NAME = f"Custom_ResNet50_{CATEGORY}"
RANDOM_SEED = 42

# Data paths
TRAIN_DIR = PROCESSED_DATA_DIR / CATEGORY / "train"
VAL_DIR = PROCESSED_DATA_DIR / CATEGORY / "val"

# HYPERPARAMETERS
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# OUTPUTS
MODEL_SAVE_PATH = MODELS_DIR / f"{CATEGORY}_best_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.txt"