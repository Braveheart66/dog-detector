# train.py
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ---- Paths ----
data_dir = "data"  # contains: data/dogs and data/not_dogs

# ---- Data generators (with augmentation + 20% validation split) ----
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# ---- Model (deeper CNN) ----
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")   # binary output
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ---- Train ----
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

# ---- Save model + class indices ----
model.save("dog_notdog_model.keras")
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("✅ Saved: dog_notdog_model.keras and class_indices.json")
