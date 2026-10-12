
from pathlib import Path
import yaml


# config.py is under src/industrial_fault_detection/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

with PARAMS_PATH.open("r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

CATEGORY = params["base"]["category"]
RANDOM_SEED = int(params["base"]["random_seed"])

train_params = params["train"]

IMAGE_SIZE = tuple(train_params["image_size"])
BATCH_SIZE = int(train_params["batch_size"])
EPOCHS = int(train_params["epochs"])
LEARNING_RATE = float(train_params["learning_rate"])
DROPOUT_RATE = float(train_params.get("dropout_rate", 0.4))
PRETRAINED_WEIGHTS = train_params.get("pretrained_weights", "imagenet")
FINE_TUNE = bool(train_params.get("fine_tune", False))

MODEL_NAME = f"ResNet50_Transfer_{CATEGORY}"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR

TRAIN_DIR = PROCESSED_DATA_DIR / CATEGORY / "train"
VAL_DIR = PROCESSED_DATA_DIR / CATEGORY / "val"

MODEL_SAVE_PATH = MODELS_DIR / f"{CATEGORY}_best_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
