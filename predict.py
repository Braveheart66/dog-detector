import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("dog_notdog_model.keras")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Invert dict so we can map prediction index → label
idx_to_class = {v: k for k, v in class_indices.items()}

# Directory with test images
test_dir = "test_images"

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)

    # Load and preprocess
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]

    pred_class = int(round(prediction))   # 0 or 1
    label = idx_to_class[pred_class]

    # Add emoji
    if label == "dogs":
        result = "Dog 🐶"
    else:
        result = "Not Dog 🙅‍♂️"

    # Confidence
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    print(f"{img_name} → {result} ({confidence:.2f}%)")
