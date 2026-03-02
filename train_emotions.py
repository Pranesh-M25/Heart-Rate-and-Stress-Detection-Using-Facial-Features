import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os

# 1. Setup Data Paths
base_dir = r"D:\Major Project\Dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# 2. Data Generators with AUGMENTATION (Crucial for accuracy)
# This creates "fake" new images by zooming/rotating existing ones to make the AI smarter
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,      # Rotate slightly
    zoom_range=0.15,        # Zoom in/out slightly
    width_shift_range=0.1,  # Shift left/right
    height_shift_range=0.1, # Shift up/down
    horizontal_flip=True,   # Flip mirror image
    fill_mode='nearest'
)

# Test data should NOT be augmented, only rescaled
test_datagen = ImageDataGenerator(rescale=1./255)

print("Loading and analyzing data...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# *** IMPORTANT: PRINT THE CLASS MAPPING ***
print("\n------------------------------------------------")
print("CHECK THIS MAPPING! Update your main code with this order:")
print(train_generator.class_indices)
print("------------------------------------------------\n")

# 3. Build a Deeper Model (VGG-style architecture)
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Block 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Block 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Block 4
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# 4. Compile with a lower learning rate for stability
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Train
print("Starting training... (This will take longer but is much better)")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,
    epochs=30,  # Increased to 30 for better learning
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 64
)

# 6. Save
model.save("my_custom_emotion_model.h5")
print("Success! Model saved.")