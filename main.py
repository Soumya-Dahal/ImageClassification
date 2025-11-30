import tensorflow as tf
from tensorflow.keras import layers, models

train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    validation_split=0.2,
    subset = 'training',
    seed=123,
    labels = "inferred",
    batch_size=16,
    image_size = (256,256),
    shuffle=True
    )
test_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/test',
    labels = "inferred",
    batch_size=16,
    image_size = (256,256)
    )
val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    validation_split=0.2,
    subset='validation',
    seed=123,
    batch_size=16,
    image_size = (256,256)
    )
#Augment data for train
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
    ])

# Configure dataset performance
AUTOTUNE = tf.data.AUTOTUNE
#data preparation
def data_preprocessing(ds, augmentation=False, prefetch=True):
    # Normalize pixel values to [0, 1]
    ds = ds.map(lambda x, y: (x/255.0, y))
    
    # Apply augmentation only to training set
    if augmentation:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    if prefetch:
        ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

#Preprocess data
data_preprocessing(train_ds, augmentation=True)
data_preprocessing(test_ds, augmentation=False)
data_preprocessing(val_ds, augmentation=False)


# Create model
model = models.Sequential([
    tf.keras.layers.Flatten(input_shape=(256, 256, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.2f}")

model.save('imgClassification.keras')
