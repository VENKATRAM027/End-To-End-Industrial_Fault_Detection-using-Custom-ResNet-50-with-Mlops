
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from industrial_fault_detection.config import (
    VAL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    MODEL_SAVE_PATH,
    MODELS_DIR,
    CATEGORY,
)


def evaluate_model():
    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_SAVE_PATH}")

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = val_ds.class_names

    # The model receives ResNet-50-preprocessed images.
    preprocess = tf.keras.applications.resnet50.preprocess_input
    val_ds = val_ds.map(
        lambda images, labels: (preprocess(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    y_true = np.concatenate(
        [labels.numpy() for _, labels in val_ds],
        axis=0,
    )
    probabilities = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    metrics = {
        "category": CATEGORY,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true)) == 2:
        # Determine the index of the defect class by name.
        defect_names = {"anomaly", "defect", "defective"}
        defect_indices = [
            i for i, name in enumerate(class_names)
            if name.lower() in defect_names
        ]
        positive_index = defect_indices[0] if defect_indices else 1

        y_true_binary = (y_true == positive_index).astype(int)
        defect_probabilities = probabilities[:, positive_index]

        metrics["roc_auc"] = float(
            roc_auc_score(y_true_binary, defect_probabilities)
        )
    else:
        metrics["roc_auc"] = None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"ResNet-50 Transfer Learning — {CATEGORY}")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    evaluate_model()
