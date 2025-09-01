import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths
train_dir = "data"

# Image generators
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode="binary",
    subset="validation"
)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save model in modern format
model.save("dog_notdog_model.keras")
print("Model saved as dog_notdog_model.keras")

# Save class indices for consistent predictions
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("Class indices saved:", train_gen.class_indices)
