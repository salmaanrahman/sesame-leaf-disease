import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "MobileNet_best_model.h5"

CLASS_NAMES = [
    "Healthy Leaf",
    "Insect Leaf Damage",
    "Leaf Spot Disease",
    "Yellowing Leaf Syndrome"
]

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Sesame Leaf Disease Detection", page_icon="🌿", layout="centered")
st.title("🌿 Sesame Leaf Disease Detection (AI Model)")
st.write("Upload a sesame leaf image to detect disease")

# ----------------------------
# Load Model (cached)
# ----------------------------
@st.cache_resource
def load_model_cached(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Keras 3 compatibility: safe_mode=False helps loading legacy .h5
    try:
        m = tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        # Older TF may not support safe_mode argument
        m = tf.keras.models.load_model(path, compile=False)

    return m

try:
    model = load_model_cached(MODEL_PATH)
except Exception as e:
    st.error("❌ Model load failed.")
    st.code(str(e))
    st.stop()

# ----------------------------
# Input size (fallback safe)
# ----------------------------
try:
    h = int(model.input_shape[1]) if model.input_shape[1] is not None else 224
    w = int(model.input_shape[2]) if model.input_shape[2] is not None else 224
except Exception:
    h, w = 224, 224

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess for MobileNet
    img_resized = img.resize((w, h))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array, verbose=0)
    idx = int(np.argmax(pred))
    conf = float(np.max(pred) * 100)

    st.success(f"✅ Prediction: **{CLASS_NAMES[idx]}**")
    st.info(f"Confidence: **{conf:.2f}%**")
