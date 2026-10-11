import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    accuracy_score
)
from pathlib import Path
import sys

# Adjusted to go up 4 levels to reach the root dir
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from industrial_fault_detection.config import (
    VAL_DIR, IMAGE_SIZE, BATCH_SIZE, 
    MODEL_SAVE_PATH, ARTIFACTS_DIR, CATEGORY
)

def evaluate_model():
    print(f"Starting Evaluation for {CATEGORY} model...")

    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_SAVE_PATH}. Run train.py first.")

    # 1. Load Validation Dataset
    # CRITICAL: shuffle=False ensures predictions align perfectly with true labels
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    class_names = val_ds.class_names

    # Extract true labels
    print("Extracting true labels...")
    y_true = np.concatenate([y for x, y in val_ds], axis=0)

    # 2. Load Model
    print("Loading model weights...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # 3. Generate Predictions
    print("Running inference...")
    y_pred_probs = model.predict(val_ds, verbose=1)
    
    # Argmax for discrete class predictions (0 or 1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Probabilities of the positive class (usually index 1) for ROC-AUC
    y_pred_positive_probs = y_pred_probs[:, 1] 

    # 4. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Try to calculate ROC-AUC. (Requires both classes to be present in val_ds)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_positive_probs)
    except ValueError:
        roc_auc = "N/A (Only one class present in validation set)"

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(report_str)
    print(f"ROC-AUC Score: {roc_auc}")
    print("="*50)

    # 5. Save Metrics to JSON
    metrics_path = ARTIFACTS_DIR / "evaluation_metrics.json"
    evaluation_results = {
        "accuracy": accuracy,
        "roc_auc": roc_auc if isinstance(roc_auc, float) else None,
        "classification_report": report_dict
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # 6. Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: Custom ResNet50 ({CATEGORY})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    print(f"Confusion matrix plot saved to {cm_path}")

if __name__ == "__main__":
    evaluate_model()