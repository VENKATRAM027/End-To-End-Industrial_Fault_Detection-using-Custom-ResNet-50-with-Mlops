
import tensorflow as tf

from industrial_fault_detection.config import (
    TRAIN_DIR,
    VAL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
)


def load_and_preprocess_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names

    if class_names != val_ds.class_names:
        raise ValueError("Train and validation class names differ.")

    # Use light augmentation only on training images.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="data_augmentation",
    )

    preprocess = tf.keras.applications.resnet50.preprocess_input

    train_ds = train_ds.map(
        lambda images, labels: (
            preprocess(augmentation(images, training=True)),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda images, labels: (preprocess(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
