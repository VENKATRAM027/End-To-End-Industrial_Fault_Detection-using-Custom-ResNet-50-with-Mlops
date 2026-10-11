import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

import mlflow
import mlflow.tensorflow
import tensorflow as tf
import tensorflow.keras

from industrial_fault_detection.config import (
    CATEGORY, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_SAVE_PATH, TRAIN_DIR, VAL_DIR, RANDOM_SEED, MODEL_NAME,
)
from industrial_fault_detection.model.resnet_50 import build_resnet50 

# DLOps: Ensure reproducibility
tensorflow.keras.utils.set_random_seed(RANDOM_SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=RANDOM_SEED),
    tf.keras.layers.RandomRotation(factor=0.08, seed=RANDOM_SEED),
    tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10, seed=RANDOM_SEED),
    tf.keras.layers.RandomContrast(factor=0.10, seed=RANDOM_SEED),
], name="data_augmentation")

def load_dataset():
    print("Loading Dataset")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels='inferred', label_mode="int",
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, labels='inferred', label_mode="int",
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False, seed=RANDOM_SEED
    )

    train_class_names = train_ds.class_names
    val_class_names = val_ds.class_names

    # Make sure both datasets have the same class mapping.
    if train_class_names != val_class_names:
        raise ValueError(f"Class names mismatch: Train {train_class_names} vs Val {val_class_names}")

    if len(train_class_names) != 2:
        raise ValueError(f"Expected 2 classes, got {len(train_class_names)}")

    train_ds = train_ds.map(
        lambda images, labels: (
            data_augmentation(images, training=True), labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # DLOps: Prevent GPU starvation
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, train_class_names

def build_model():
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    model = build_resnet50(
        input_shape=input_shape,
        num_classes=2,
        dropout_rate=0.2,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model

def train_model():
    experiment_name = f"Fault_Detection_{CATEGORY}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.log_params({
            'category': CATEGORY,
            "model_name": MODEL_NAME,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'augmentation': True,
        })

        (train_ds, val_ds, class_names) = load_dataset()
        mlflow.log_param("class_0", class_names[0])
        mlflow.log_param('class_1', class_names[1])

        model = build_model()
        model.summary()

        total_parameters = model.count_params()
        mlflow.log_param('total_parameters', total_parameters)

        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(MODEL_SAVE_PATH), monitor='val_loss',
                mode='min', save_best_only=True, verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', mode='min', patience=5,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', mode='min', factor=0.2,
                patience=2, min_lr=1e-7, verbose=1,
            ),    
        ]

        history = model.fit(
            train_ds, validation_data=val_ds,
            epochs=EPOCHS, callbacks=callbacks,
        )

        best_val_loss = min(history.history['val_loss'])
        best_val_accuracy = max(history.history['val_accuracy'])
        best_epoch = history.history['val_loss'].index(best_val_loss) + 1 

        mlflow.log_metric('best_val_accuracy', float(best_val_accuracy))
        mlflow.log_metric('best_val_loss', float(best_val_loss))
        mlflow.log_param('best_epoch', best_epoch)

        evaluation_results = model.evaluate(val_ds, verbose=1, return_dict=True)

        for metric_name, metric_value in evaluation_results.items():
            mlflow.log_metric(f"final_{metric_name}", float(metric_value))

        # Save class mappings as an artifact
        class_names_path = MODEL_SAVE_PATH.parent / "class_names.txt"
        with open(class_names_path, "w", encoding="utf-8") as file:
            for class_name in class_names:
                file.write(f"{class_name}\n")
        
        mlflow.log_artifact(str(class_names_path))
        mlflow.tensorflow.log_model(model, artifact_path="model")

if __name__ == "__main__":
    train_model()