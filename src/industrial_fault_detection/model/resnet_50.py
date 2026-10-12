import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def build_resnet50(input_shape=(224, 224, 3), num_classes=2, dropout_rate=0.4):
    """
    Builds a Transfer Learning model using a pre-trained ResNet-50 base.
    """
    # 1. Load the pre-trained ImageNet ResNet50 (excluding the top classification layers)
    base_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # 2. Freeze the base model layers so they don't lose their learned features during early training
    base_model.trainable = False

    # 3. Build our custom classification head on top
    inputs = layers.Input(shape=input_shape, name='image')
    
    # Keras application models expect standard 0-255 image inputs natively.
    # Pass inputs directly into the frozen base model.
    x = base_model(inputs, training=False)
    
    # Add classification head
    x = layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
    x = layers.Dropout(rate=dropout_rate, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x)

    model = Model(inputs=inputs, outputs=outputs, name='industrial_fault_transfer_resnet50')
    return model

if __name__ == "__main__":
    # Quick test to ensure it compiles
    model = build_resnet50()
    model.summary()