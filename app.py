import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown
from keras.models import load_model
import os

# Path where model will be saved
MODEL_PATH = "best_dog_notdog_model.keras"

# Google Drive file ID
FILE_ID = "1Ojj5UmyvafwRPrNnJMN-Ud6kUd1BRz-Z"  # replace with your file ID
URL = f"https://drive.google.com/file/d/1Ojj5UmyvafwRPrNnJMN-Ud6kUd1BRz-Z/view?usp=sharing"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    gdown.download(URL, MODEL_PATH, quiet=False)

# -------------------------------
# 1. Load model
# -------------------------------
model = load_model(MODEL_PATH)  # Update path if needed

# -------------------------------
# 2. Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="Dog vs Not a dog Classifier", layout="centered")
st.title("🐶 Dog vs Not a dog Classifier")
st.markdown("Upload one or multiple images of only dogs and cats, and the model will predict whether each image is a Dog or Not a Dog (Cat).")

# -------------------------------
# 3. File Uploader
# -------------------------------
uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def preprocess_image(img, target_size=(128, 128)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# 4. Process & Predict
# -------------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write("---")
        st.subheader(uploaded_file.name)
        img = Image.open(uploaded_file)
        st.image(img, width="stretch")

        # Preprocess
        img_array = preprocess_image(img)

       # Define class labels
        class_labels = {0: "Dog 🐶", 1: "Not a dog cuhh 😼"}

       # Predict
        prediction = model.predict(img_array)[0][0]
        pred_class = 0 if prediction < 0.5 else 1
        confidence = prediction if pred_class == 1 - prediction else 1 

        # Label
        label = f"{class_labels[pred_class]}"
        conf_percent = confidence * 100

         # Display
        st.markdown(f"*Prediction:* {label} ({conf_percent:.2f}% confidence)")
        st.progress(int(conf_percent))



# -------------------------------
# 5. Footer
# -------------------------------
st.write("---")
st.write("Model trained using TensorFlow & Keras | Streamlit Web App |")  
