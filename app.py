"""Streamlit front end for the TB chest X-ray classifier."""

import streamlit as st
from PIL import Image

import inference
import models
from config import ARCHITECTURES

st.set_page_config(page_title="TB Classifier", layout="wide")

st.title("🫁 Tuberculosis Detection from Chest X-ray")
st.markdown(
    "Upload a chest X-ray image and the model will predict whether it shows "
    "signs of **Tuberculosis (TB)**."
)
st.divider()

st.markdown("## 📤 Input")
col_model, col_upload = st.columns([1, 1])
with col_model:
    architecture = st.selectbox("Choose a model", ARCHITECTURES)
with col_upload:
    uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload a chest X-ray image to begin.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

with st.spinner(f"Loading {architecture}…"):
    model, target_layer = models.load_classifier("oversampling", architecture)

with st.spinner("Analyzing image…"):
    tensor = inference.preprocess(image, architecture)
    label, confidence, index = inference.classify(model, tensor)
    display = inference.as_display_array(image, architecture)
    overlay = inference.blend_cam(
        display, inference.grad_cam(model, target_layer, tensor, index)
    )

st.divider()
st.markdown("## 🧠 Prediction")
if label == "TB":
    st.error(f"⚠️ **TB Detected** — Confidence: {confidence:.2%}")
else:
    st.success(f"✅ **Normal** — Confidence: {confidence:.2%}")

st.divider()
st.markdown("## 🔍 Model Explanation")
left, right = st.columns([1, 1])
with left:
    st.markdown("### Original Chest X-ray")
    st.image(image, use_container_width=True)
with right:
    st.markdown("### Grad-CAM Heatmap")
    st.image(overlay, use_container_width=True)
