import streamlit as st
import torch
from PIL import Image
from utils import load_model, predict_and_explain

# -------------------- CONFIGURATION --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model registry – extract file ID from Google Drive links
# Example: https://drive.google.com/file/d/1s_NHkelNY1PouNKsGZvrMLAMtbXBAOrA/view → ID = 1s_NHkelNY1PouNKsGZvrMLAMtbXBAOrA
MODEL_CONFIGS = {
    "EfficientNetB3": {
        "url": "https://drive.google.com/uc?id=1s_NHkelNY1PouNKsGZvrMLAMtbXBAOrA",   # direct download link
        "filename": "models/efficientnet.pt",
        "needs_norm": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": 224
    },
    "Custom_CNN": {
        "url": "https://drive.google.com/uc?id=1RuqyYxoriMQnynEzDix88kW5aKWUJlJ7",
        "filename": "models/custom_cnn.pt",
        "needs_norm": False,
        "mean": None,
        "std": None,
        "input_size": 224
    },
    "DenseNet169": {
        "url": "https://drive.google.com/uc?id=12i2wPlarZMb_T1OjFOeXKknvLP3CygT7",
        "filename": "models/densenet.pt",
        "needs_norm": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": 224
    },
    "ResNet50": {
        "url": "https://drive.google.com/uc?id=1f-jG2999u7h3swXy1nKm_Zh9kscoejlG",
        "filename": "models/resnet.pt",
        "needs_norm": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": 224
    },
    "VGG16": {
        "url": "https://drive.google.com/uc?id=1H0l0DB3e8bGn7FOUc3a41vjD9HSh0cfX",
        "filename": "models/vgg.pt",
        "needs_norm": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": 224
    }
}

CLASS_NAMES = ["Normal", "TB"]   # index 0 = Normal, 1 = TB

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="TB Classifier", layout="wide")
st.title("🫁 Tuberculosis Detection from Chest X‑ray")
st.markdown("Upload a chest X‑ray image, and the model will predict whether it shows signs of TB.")

# Model selection
selected_model = st.selectbox("Choose a model", list(MODEL_CONFIGS.keys()))

# Load model (cached)
model, target_layer, config = load_model(selected_model, MODEL_CONFIGS, device)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        pred_name, confidence, cam_image, _ = predict_and_explain(
            model, target_layer, image, config, device, CLASS_NAMES
        )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", pred_name)
        st.metric("Confidence", f"{confidence:.2%}")
    with col2:
        st.image(cam_image, caption="Grad‑CAM Explanation", use_column_width=True)