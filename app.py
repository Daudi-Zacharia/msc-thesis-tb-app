"""Streamlit front end for the TB chest X-ray classification/segmentation tool."""

import io
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import download
import inference
import models
from config import (
    ARCHITECTURES,
    CHECKPOINT_FILENAMES,
    MODES,
    SEGMENTATION_CHECKPOINT,
)

SAMPLE_ROOT = Path("sample_images")

st.set_page_config(page_title="TB Chest X-ray Analyser", layout="wide")


# -------------------- cached loaders --------------------

def get_classifier(scheme, architecture):
    """Load a classifier, reusing the session's copy if it is already loaded."""
    cache = st.session_state.setdefault("classifiers", {})
    key = (scheme, architecture)
    if key in cache:
        return cache[key]

    filename = CHECKPOINT_FILENAMES[scheme][architecture]
    if download.is_cached(filename):
        message = f"Loading {architecture} ({scheme}) weights…"
    else:
        message = f"Downloading {architecture} ({scheme}) weights…"

    with st.spinner(message):
        cache[key] = models.load_classifier(scheme, architecture)
    return cache[key]


def get_segmenter():
    if "segmenter" in st.session_state:
        return st.session_state["segmenter"]

    if download.is_cached(SEGMENTATION_CHECKPOINT):
        message = "Loading U-Net segmentation weights…"
    else:
        message = "Downloading U-Net segmentation weights…"

    with st.spinner(message):
        st.session_state["segmenter"] = models.load_segmenter()
    return st.session_state["segmenter"]


# -------------------- helpers --------------------

def list_samples():
    """(label, path) pairs for the bundled example X-rays."""
    samples = []
    for group in ("tb", "normal"):
        folder = SAMPLE_ROOT / group
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                samples.append((f"{group.upper()} — {path.name}", str(path)))
    return samples


def to_png_bytes(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def analyse(image, mode_config, architecture):
    """Run the pipeline for the selected mode and return everything to render.

    Grad-CAM is computed from whatever tensor the model actually receives, but
    is always rendered over the original X-ray — a heatmap on a mostly-black
    masked image is very hard to read.
    """
    result = {"original": image, "segmented": None, "prediction": None}

    model_input = image
    if mode_config["segment"]:
        unet = get_segmenter()
        with st.spinner("Segmenting lungs…"):
            segmented, _ = inference.segment(unet, image)
        result["segmented"] = segmented
        model_input = segmented

    if mode_config["classify"]:
        scheme = mode_config["scheme"]
        model, target_layer = get_classifier(scheme, architecture)
        with st.spinner("Running inference…"):
            tensor = inference.preprocess(model_input, architecture)
            label, confidence, index = inference.classify(model, tensor)
            display = inference.as_display_array(image, architecture)
            grayscale_cam = inference.grad_cam(model, target_layer, tensor, index)
        result["prediction"] = {
            "label": label,
            "confidence": confidence,
            "cam": grayscale_cam,
            "display": display,
        }

    return result


# -------------------- sidebar --------------------

st.sidebar.title("Controls")

mode = st.sidebar.radio("Mode", list(MODES.keys()))
mode_config = MODES[mode]

architecture = None
if mode_config["classify"]:
    architecture = st.sidebar.selectbox("Classification model", ARCHITECTURES)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload a chest X-ray", type=["png", "jpg", "jpeg"]
)

samples = list_samples()
sample_path = None
if samples:
    labels = ["—"] + [label for label, _ in samples]
    choice = st.sidebar.selectbox("…or try an example", labels)
    if choice != "—":
        sample_path = dict((label, path) for label, path in samples)[choice]

opacity = 0.5
if mode_config["classify"]:
    st.sidebar.divider()
    opacity = st.sidebar.slider(
        "Grad-CAM overlay opacity", 0, 100, 50,
        help="Blend between the original X-ray (0%) and the heatmap (100%).",
    ) / 100.0


# -------------------- header --------------------

st.title("🫁 TB Chest X-ray Analyser")
st.markdown(
    "Segment the lung fields, classify a chest X-ray as **Normal** or "
    "**Tuberculosis**, or do both — with a Grad-CAM heatmap showing which "
    "regions drove the prediction."
)
st.warning(
    "Research prototype — not a diagnostic tool. "
    "Do not use for clinical decision-making."
)
st.divider()


# -------------------- resolve input --------------------

image = None
source_key = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    source_key = ("upload", uploaded_file.name, uploaded_file.size)
elif sample_path is not None:
    image = Image.open(sample_path).convert("RGB")
    source_key = ("sample", sample_path, os.path.getsize(sample_path))

if image is None:
    st.info(
        "Upload a chest X-ray in the sidebar, or pick one of the bundled "
        "examples, to get started."
    )
    st.stop()


# -------------------- run (or reuse a cached run) --------------------

# The opacity slider only changes how the heatmap is blended, so keying the
# cached run on (image, mode, model) lets the slider re-render instantly
# instead of re-running segmentation and inference.
signature = (source_key, mode, architecture)
if st.session_state.get("run_signature") != signature:
    st.session_state["run_result"] = analyse(image, mode_config, architecture)
    st.session_state["run_signature"] = signature

result = st.session_state["run_result"]


# -------------------- results --------------------

if result["segmented"] is not None:
    st.subheader("Segmentation")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original**")
        st.image(result["original"], use_container_width=True)
    with right:
        st.markdown("**Segmented lungs**")
        st.image(result["segmented"], use_container_width=True)

    st.download_button(
        "Download segmented image",
        data=to_png_bytes(result["segmented"]),
        file_name="segmented.png",
        mime="image/png",
    )

prediction = result["prediction"]
if prediction is not None:
    st.divider()
    st.subheader("Classification")

    label = prediction["label"]
    confidence = prediction["confidence"]

    if label == "TB":
        st.error(f"### ⚠️ {label}")
    else:
        st.success(f"### ✅ {label}")

    st.markdown(f"**Confidence: {confidence:.1%}**")
    st.progress(confidence)
    st.caption(
        f"{architecture} · {mode_config['scheme']} checkpoint · "
        f"{'segmented' if mode_config['segment'] else 'unsegmented'} input"
    )

    st.divider()
    st.subheader("Grad-CAM")
    if mode_config["segment"]:
        st.caption(
            "Computed on the segmented image the model saw, drawn over the "
            "original X-ray for readability."
        )

    overlay = inference.blend_cam(prediction["display"], prediction["cam"], opacity)

    left, right = st.columns(2)
    with left:
        st.markdown("**Original**")
        st.image(result["original"], use_container_width=True)
    with right:
        st.markdown("**Grad-CAM overlay**")
        st.image(overlay, use_container_width=True)

    st.download_button(
        "Download Grad-CAM overlay",
        data=to_png_bytes(overlay),
        file_name="gradcam.png",
        mime="image/png",
    )
