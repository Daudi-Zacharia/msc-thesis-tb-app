import os
import torch
import torch.nn as nn
from torchvision import transforms as T
import gdown
import streamlit as st
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -------------------- CUSTOM CNN ARCHITECTURE (for state dict loading) --------------------
class CustomCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),

            nn.Flatten(),
            nn.Linear(43264, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.layers(x)

# -------------------- HELPER FUNCTIONS --------------------
def find_last_conv(model):
    """Find the last convolutional layer for Grad‑CAM."""
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d found")
    return last_conv

@st.cache_resource
def load_model(model_key, model_configs, device):
    """
    Download (if needed) and load the model.
    Returns (model, target_layer, config)
    """
    config = model_configs[model_key]
    os.makedirs(os.path.dirname(config["filename"]), exist_ok=True)

    # Download if file not present
    if not os.path.exists(config["filename"]):
        with st.spinner(f"Downloading {model_key} model..."):
            # gdown expects the file ID, not full URL
            # You can pass the full URL, but gdown works with either.
            gdown.download(config["url"], config["filename"], quiet=False)

    # Load model – handle both full model and state dict
    if model_key == "Custom_CNN":
        # Custom CNN was saved as state dict
        model = CustomCNN().to(device)
        state_dict = torch.load(config["filename"], map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Other models were saved as full models
        model = torch.load(config["filename"], map_location=device, weights_only=False)
        model = model.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    target_layer = find_last_conv(model)
    return model, target_layer, config

def predict_and_explain(model, target_layer, image_pil, config, device, class_names):
    """
    Run inference and return (pred_name, confidence, cam_image, preprocessed_image)
    """
    # Preprocess
    transform_list = [
        T.Resize((config["input_size"], config["input_size"])),
        T.ToTensor()
    ]
    if config["needs_norm"]:
        transform_list.append(T.Normalize(mean=config["mean"], std=config["std"]))
    transform = T.Compose(transform_list)
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    pred_name = class_names[pred_class]

    # Grad‑CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Denormalise for display
    if config["needs_norm"]:
        mean = torch.tensor(config["mean"]).view(3,1,1)
        std = torch.tensor(config["std"]).view(3,1,1)
        img_denorm = input_tensor.squeeze(0).cpu() * std + mean
    else:
        img_denorm = input_tensor.squeeze(0).cpu()
    img_denorm = img_denorm.clamp(0, 1).permute(1,2,0).numpy()

    cam_image = show_cam_on_image(img_denorm, grayscale_cam, use_rgb=True)
    return pred_name, confidence, cam_image, img_denorm