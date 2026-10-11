import tensorflow as tf
from industrial_fault_detection.config import TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=RANDOM_SEED), 
    tf.keras.layers.RandomRotation(factor=0.08, seed=RANDOM_SEED), 
    tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10, seed=RANDOM_SEED), 
    tf.keras.layers.RandomContrast(factor=0.10, seed=RANDOM_SEED), #[cite: 2]
], name="data_augmentation")

def load_and_preprocess_data():
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

    if train_class_names != val_class_names:
        raise ValueError("Class mismatch between train and validation sets.") 

    train_ds = train_ds.map(
        lambda images, labels: (data_augmentation(images, training=True), labels), 
        num_parallel_calls=tf.data.AUTOTUNE, 
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE) 
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE) 

    return train_ds, val_ds, train_class_names 