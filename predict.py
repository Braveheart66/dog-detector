import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("best_dog_notdog_model.keras")

# Keep mapping consistent with training
class_labels = {1: "Dog 🐶", 0: "Not Dog 🙅‍♂️"}

def predict_and_plot(img_path):
    # Preprocess image (match MobileNetV2 input: 128x128)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    pred_class = 0 if prediction < 0.5 else 1
    confidence = prediction if pred_class == 1 - prediction else 1

    # Label
    label = f"{class_labels[pred_class]} ({confidence*100:.2f}%)"

    # Show image with label
    plt.imshow(image.load_img(img_path))
    plt.title(label)
    plt.axis("off")
    plt.show()

# Example test
predict_and_plot(r"C:\Users\shiva\OneDrive\Documents\dog-detector\test_images\test_cat1.jpg")
predict_and_plot(r"C:\Users\shiva\OneDrive\Documents\dog-detector\test_images\test_dog1.jpg")
