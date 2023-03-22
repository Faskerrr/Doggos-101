import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.applications.resnet50 as resnet50
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def init_model(input_shape: tuple) -> Model:
    """
    Initialize a ResNet50 model with a custom head.
    Args:
        input_shape: shape of the input images
    Returns:
        model: the initialized model
    """
    # Load the ResNet50 model without the head
    base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    # Freeze the layers of the ResNet50 model
    base_model.trainable = False
    # Create the model
    model = tf.keras.Sequential([
        layers.Input((224, 224, 3)),
        layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(120, activation='softmax')
    ])
    print("✅ model initialized")
    return model

def compile_model(model: Model) -> Model:
    """
    Compile the model.
    Args:
        model: the model to compile
    Returns:
        model: the compiled model
    """
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ model compiled")
    return model

def train_model(model: Model,
                train_ds: Dataset,
                validation_ds: Dataset,
                epochs: int):
    """
    Train the model.
    Args:
        model: the model to train
        train_ds: the training dataset
        validation_ds: the validation dataset
        epochs: number of epochs
    """
    # Create callbacks
    ## Define EarlyStopping, ModelCheckpoint and ReduceLROnPlateau callbacks
    es = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
    ## Create a list of callbacks
    callbacks = [es, mc, rlr]
    # Train the model
    history = model.fit(train_ds,
                        validation_data=validation_ds,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=1)

    print("✅ model trained")
    return model, history
