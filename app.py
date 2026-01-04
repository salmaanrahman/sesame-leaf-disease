import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "MobileNet_best_model.h5"

st.title("🌿 Sesame Leaf Disease Detection (AI Model)")
st.write("Upload a sesame leaf image to detect disease")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please upload MobileNet_best_model.h5 in repo root.")
    st.stop()

@st.cache_resource
def load_model():
    # Keras2 compatible load
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

h, w = model.input_shape[1], model.input_shape[2]

CLASS_NAMES = [
    "Healthy Leaf",
    "Insect Leaf Damage",
    "Leaf Spot Disease",
    "Yellowing Leaf Syndrome"
]

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((w, h))
    x = np.array(img, dtype=np.float32)

    # ✅ IMPORTANT: training এ যেটা ছিল সেটাই use করতে হবে
    # MobileNet training যদি preprocess_input দিয়ে হয়ে থাকে:
    x = tf.keras.applications.mobilenet.preprocess_input(x)

    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    idx = int(np.argmax(pred))
    conf = float(np.max(pred) * 100)

    st.success(f"✅ Prediction: **{CLASS_NAMES[idx]}**")
    st.info(f"Confidence: **{conf:.2f}%**")
