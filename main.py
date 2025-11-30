import tensorflow as tf

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/kaggle/input/intel-image-classification/seg_train/seg_train',
    validation_split=0.2,
    subset='training',
    seed=123,
    labels="inferred",
    batch_size=16,
    image_size=(256, 256),
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    '/kaggle/input/intel-image-classification/seg_train/seg_train',
    validation_split=0.2,
    subset='validation',
    seed=123,
    batch_size=16,
    image_size=(256, 256)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    '/kaggle/input/intel-image-classification/seg_test/seg_test',
    labels="inferred",
    batch_size=16,
    image_size=(256, 256)
)

# Augment data for train
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

# Configure dataset performance
AUTOTUNE = tf.data.AUTOTUNE

# Data preparation
def prepare_dataset(ds, augmentation=False, prefetch=True):
    # Normalize pixel values to [0, 1]
    ds = ds.map(lambda x, y: (x/255.0, y))
    
    # Apply augmentation only to training set
    if augmentation:
        # FIX: Use map correctly for augmentation
        ds = ds.map(lambda x, y: (data_augmentation(x), y))
    
    if prefetch:
        ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

# prepare datasets 
train_ds_processed = prepare_dataset(train_ds, augmentation=True)
val_ds_processed = prepare_dataset(val_ds, augmentation=False)
test_ds_processed = prepare_dataset(test_ds, augmentation=False)

# Create model
model = tf.keras.Sequential([
    
    # First Convolutional Block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Second Convolutional Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Third Convolutional Block
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Fourth Convolutional Block
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),  # Increased to 256
    tf.keras.layers.MaxPooling2D(2, 2),

    # Global Average Pooling instead of Flatten (better for overfitting)
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),

    # Final Dense layers for classification
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),  # Added for stability
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Explicit optimizer
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Add callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )
]

# Train model
history = model.fit(
    train_ds_processed,  
    validation_data=val_ds_processed,
    epochs=30,
    callbacks=callbacks
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_ds_processed)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

model.save('imgClassification.keras')