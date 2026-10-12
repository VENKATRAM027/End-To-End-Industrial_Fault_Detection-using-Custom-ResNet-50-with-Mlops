import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

import mlflow
import mlflow.tensorflow
import tensorflow as tf

from industrial_fault_detection.config import (
    CATEGORY, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_SAVE_PATH, TRAIN_DIR, VAL_DIR, RANDOM_SEED, MODEL_NAME,
)
from industrial_fault_detection.model.resnet_50 import build_resnet50 
from industrial_fault_detection.data.data_preprocessing import load_and_preprocess_data

tf.keras.utils.keras.set_random_seed(RANDOM_SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

def get_class_weights():
    """Dynamically calculates class weights to combat MVTec AD imbalance."""
    anomaly_count = len(list((TRAIN_DIR / "anomaly").glob("*.png")))
    good_count = len(list((TRAIN_DIR / "good").glob("*.png")))
    total = anomaly_count + good_count

    # Class 0 = anomaly, Class 1 = good (Alphabetical ordering from tf.data)
    weight_0 = (1 / anomaly_count) * (total / 2.0)
    weight_1 = (1 / good_count) * (total / 2.0)

    print(f"Class Weights Calculated - Anomaly (0): {weight_0:.2f}, Good (1): {weight_1:.2f}")
    return {0: weight_0, 1: weight_1}

def build_model():
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    model = build_resnet50(
        input_shape=input_shape,
        num_classes=2,
        dropout_rate=0.4, # Increased dropout slightly to prevent overfitting on minority class
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
        class_weights = get_class_weights()
        
        mlflow.log_params({
            'category': CATEGORY,
            "model_name": MODEL_NAME,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_anomaly': class_weights[0],
            'weight_good': class_weights[1]
        })

        # Load data using your separated preprocessing script
        (train_ds, val_ds, class_names) = load_and_preprocess_data()
        
        mlflow.log_param("class_0", class_names[0])
        mlflow.log_param('class_1', class_names[1])

        model = build_model()
        mlflow.log_param('total_parameters', model.count_params())

        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(MODEL_SAVE_PATH), monitor='val_loss',
                mode='min', save_best_only=True, verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', mode='min', patience=8, # Increased patience for weighted learning
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', mode='min', factor=0.2,
                patience=3, min_lr=1e-7, verbose=1,
            ),    
        ]

        # Injecting the class weights into the training loop
        history = model.fit(
            train_ds, 
            validation_data=val_ds,
            epochs=EPOCHS, 
            callbacks=callbacks,
            class_weight=class_weights
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

        class_names_path = MODEL_SAVE_PATH.parent / "class_names.txt"
        with open(class_names_path, "w", encoding="utf-8") as file:
            for class_name in class_names:
                file.write(f"{class_name}\n")
        
        mlflow.log_artifact(str(class_names_path))
        mlflow.tensorflow.log_model(model, artifact_path="model")

if __name__ == "__main__":
    train_model()