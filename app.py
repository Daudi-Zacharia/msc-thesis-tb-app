"""Streamlit front end for the TB chest X-ray classification/segmentation tool."""

import streamlit as st
from PIL import Image

import inference
import models
from config import ARCHITECTURES, MODES

st.set_page_config(page_title="TB Chest X-ray Analyser", layout="wide")

st.title("🫁 TB Chest X-ray Analyser")
st.divider()

mode = st.radio("Mode", list(MODES.keys()))
mode_config = MODES[mode]

architecture = None
if mode_config["classify"]:
    architecture = st.selectbox("Classification model", ARCHITECTURES)

uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Please upload a chest X-ray image to begin.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

# In "Segment then classify" the classifier sees the segmented image, which is
# also what the matching (segmented-scheme) checkpoints were trained on.
model_input = image
segmented = None
if mode_config["segment"]:
    with st.spinner("Loading U-Net…"):
        unet = models.load_segmenter()
    with st.spinner("Segmenting lungs…"):
        segmented, _ = inference.segment(unet, image)
    model_input = segmented

    st.divider()
    st.markdown("## Segmentation")
    st.image(segmented, use_container_width=True)

if mode_config["classify"]:
    scheme = mode_config["scheme"]
    with st.spinner(f"Loading {architecture} ({scheme})…"):
        model, target_layer = models.load_classifier(scheme, architecture)

    with st.spinner("Running inference…"):
        tensor = inference.preprocess(model_input, architecture)
        label, confidence, index = inference.classify(model, tensor)
        # Grad-CAM comes from the tensor the model saw, but is drawn over the
        # original X-ray — a masked image is mostly black and unreadable.
        display = inference.as_display_array(image, architecture)
        overlay = inference.blend_cam(
            display, inference.grad_cam(model, target_layer, tensor, index)
        )

    st.divider()
    st.markdown("## Prediction")
    if label == "TB":
        st.error(f"⚠️ **TB Detected** — Confidence: {confidence:.2%}")
    else:
        st.success(f"✅ **Normal** — Confidence: {confidence:.2%}")

    st.markdown("## Grad-CAM")
    st.image(overlay, use_container_width=True)
