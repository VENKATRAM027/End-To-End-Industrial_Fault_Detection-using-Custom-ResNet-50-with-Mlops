
import json
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from industrial_fault_detection.config import (
    CATEGORY,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    TRAIN_DIR,
    RANDOM_SEED,
    MODEL_NAME,
    MODELS_DIR,
)
from industrial_fault_detection.model.resnet_50 import (
    build_resnet50,
    compile_model,
)
from industrial_fault_detection.data.data_preprocessing import (
    load_and_preprocess_data,
)


def get_class_weights(class_names):
    counts = [
        len(list((TRAIN_DIR / name).glob("*.png")))
        for name in class_names
    ]

    if any(count == 0 for count in counts):
        raise ValueError(f"Empty class detected: {dict(zip(class_names, counts))}")

    total = sum(counts)
    n_classes = len(counts)

    return {
        index: total / (n_classes * count)
        for index, count in enumerate(counts)
    }


def train_model():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    train_ds, val_ds, class_names = load_and_preprocess_data()
    class_weights = get_class_weights(class_names)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(f"Fault_Detection_{CATEGORY}")

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.log_params(
            {
                "category": CATEGORY,
                "architecture": "ResNet50_ImageNet_TransferLearning",
                "weights": "imagenet",
                "backbone_frozen": True,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "random_seed": RANDOM_SEED,
                "class_names": ",".join(class_names),
            }
        )

        for index, weight in class_weights.items():
            mlflow.log_param(f"class_weight_{class_names[index]}", weight)

        model = build_resnet50(
            input_shape=(224, 224, 3),
            num_classes=len(class_names),
            dropout_rate=0.4,
            weights="imagenet",
            fine_tune=False,
        )

        compile_model(model, learning_rate=LEARNING_RATE)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(MODEL_SAVE_PATH),
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                mode="min",
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=2,
                factor=0.2,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
        )

        # Reload the best checkpoint, not the final in-memory epoch.
        best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

        metrics = best_model.evaluate(
            val_ds,
            return_dict=True,
            verbose=1,
        )

        for name, value in metrics.items():
            mlflow.log_metric(f"best_checkpoint_{name}", float(value))

        best_epoch = min(
            range(len(history.history["val_loss"])),
            key=lambda i: history.history["val_loss"][i],
        ) + 1

        mlflow.log_param("best_epoch", best_epoch)

        class_names_path = MODELS_DIR / "class_names.json"
        class_names_path.write_text(
            json.dumps(class_names, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(class_names_path))
        mlflow.tensorflow.log_model(best_model, artifact_path="model")

        print(f"Best model saved to: {MODEL_SAVE_PATH}")
        print(f"Class names: {class_names}")
        print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    train_model()
