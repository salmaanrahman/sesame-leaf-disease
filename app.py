import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet import preprocess_input

MODEL_PATH = "MobileNet_best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Healthy Leaf",
    "Insect Leaf Damage",
    "Leaf Spot Disease",
    "Yellowing Leaf Syndrome"
]


st.title("🌿 Sesame Leaf Disease Detection (AI Model)")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    IMG_SIZE = (224, 224)
    img = img.resize(IMG_SIZE)

    img_array = preprocess_input(np.array(img))   # FIXED
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"### 🟢 Prediction: **{CLASS_NAMES[result_index]}**")
    st.info(f"### 🔍 Confidence: **{confidence:.2f}%**")
