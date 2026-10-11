import tensorflow as tf
from tensorflow.keras import layers, Model



def identity_block(x, filters, name=None): 
    f1, f2, f3 = filters
    shortcut = x

    # 1 x 1 convolution
    x = layers.Conv2D(f1, kernel_size=1, strides=1, padding='valid', use_bias=False, name=None if name is None else f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn1")(x)
    x = layers.Activation('relu', name=None if name is None else f"{name}_relu1")(x)

    # 3 x 3 convolution
    x = layers.Conv2D(f2, kernel_size=3, strides=1, padding='same', use_bias=False, name=None if name is None else f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(x)
    x = layers.Activation('relu', name=None if name is None else f"{name}_relu2")(x)

    # 1 x 1 convolution
    x = layers.Conv2D(f3, kernel_size=1, strides=1, padding='valid', use_bias=False, name=None if name is None else f"{name}_conv3")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn3")(x)

    # Residual connection
    x = layers.Add(name=None if name is None else f"{name}_add")([x, shortcut])
    x = layers.Activation('relu', name=None if name is None else f"{name}_output")(x)
    
    return x 


def conv_block(x, filters, stride=2, name=None):
    f1, f2, f3 = filters
    shortcut = x

    x = layers.Conv2D(f1, kernel_size=1, strides=stride, padding='valid', use_bias=False, name=None if name is None else f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn1")(x)
    x = layers.Activation('relu', name=None if name is None else f"{name}_relu1")(x)

    x = layers.Conv2D(f2, kernel_size=3, strides=1, padding='same', use_bias=False, name=None if name is None else f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(x)
    x = layers.Activation('relu', name=None if name is None else f"{name}_relu2")(x)

    x = layers.Conv2D(f3, kernel_size=1, strides=1, padding='valid', use_bias=False, name=None if name is None else f"{name}_conv3")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn3")(x)

    # shortcut path
    shortcut = layers.Conv2D(f3, kernel_size=1, strides=stride, padding='valid', use_bias=False, name=None if name is None else f"{name}_shortcut_conv")(shortcut)
    shortcut = layers.BatchNormalization(name=None if name is None else f"{name}_shortcut_bn")(shortcut) 

    # Residual connection
    x = layers.Add(name=None if name is None else f"{name}_add")([x, shortcut])
    x = layers.Activation('relu', name=None if name is None else f"{name}_output")(x)
    
    return x 


def build_resnet50(input_shape=(224, 224, 3), num_classes=2, dropout_rate=0.2):
    inputs = layers.Input(shape=input_shape, name='image')    

    # Preprocessing
    x = layers.Rescaling(scale=1.0/255.0, name='rescaling')(inputs)

    # Stem
    x = layers.ZeroPadding2D(padding=3, name='stem_padding')(x)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='valid', use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('relu', name='stem_relu')(x)
    x = layers.ZeroPadding2D(padding=1, name='stem_pool_padding')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='stem_pool')(x)

    # Stage 2
    x = conv_block(x, filters=[64, 64, 256], stride=1, name="stage2_block1")
    x = identity_block(x, filters=[64, 64, 256], name="stage2_block2")
    x = identity_block(x, filters=[64, 64, 256], name="stage2_block3")

    # Stage 3
    x = conv_block(x, filters=[128, 128, 512], stride=2, name="stage3_block1")
    x = identity_block(x, filters=[128, 128, 512], name="stage3_block2")
    x = identity_block(x, filters=[128, 128, 512], name="stage3_block3")
    x = identity_block(x, filters=[128, 128, 512], name="stage3_block4")

    # Stage 4
    x = conv_block(x, filters=[256, 256, 1024], stride=2, name="stage4_block1")
    x = identity_block(x, filters=[256, 256, 1024], name="stage4_block2")
    x = identity_block(x, filters=[256, 256, 1024], name="stage4_block3")
    x = identity_block(x, filters=[256, 256, 1024], name="stage4_block4")
    x = identity_block(x, filters=[256, 256, 1024], name="stage4_block5")
    x = identity_block(x, filters=[256, 256, 1024], name="stage4_block6")

    # Stage 5
    x = conv_block(x, filters=[512, 512, 2048], stride=2, name="stage5_block1")
    x = identity_block(x, filters=[512, 512, 2048], name="stage5_block2")
    x = identity_block(x, filters=[512, 512, 2048], name="stage5_block3")

    # Classification Head
    x = layers.GlobalAveragePooling2D(name='global_average_pooling')(x) 
    x = layers.Dropout(rate=dropout_rate, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x) 

    model = Model(inputs=inputs, outputs=outputs, name='industrial_fault_resnet50')
    return model

def compile_model(num_classes=2, learning_rate=1e-4):    
    model = build_resnet50(input_shape=(224, 224, 3), num_classes=num_classes, dropout_rate=0.2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model

def get_callbacks():
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='artifacts/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=5, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=2, min_lr=1e-7, verbose=1)
    ]
    return callbacks 

if __name__ == "__main__":
    model = compile_model()

    model.summary()

    dummy_image = tf.random.uniform(
        shape=(1, 224, 224, 3)
    )

    prediction = model.predict(
        dummy_image,
        verbose=0
    )


    print("\nPrediction shape:", prediction.shape)
    print("Prediction:", prediction)