"""Preprocessing, prediction and Grad-CAM."""

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms as T

from config import CLASS_NAMES, PREPROCESSING

DEVICE = torch.device("cpu")  # inference is single-image and CPU-only


def build_transform(architecture):
    config = PREPROCESSING[architecture]
    steps = [T.Resize((config["input_size"], config["input_size"])), T.ToTensor()]
    if config["mean"] is not None:
        steps.append(T.Normalize(mean=config["mean"], std=config["std"]))
    return T.Compose(steps)


def preprocess(image_pil, architecture):
    """PIL image → normalised (1, 3, S, S) tensor for the given architecture."""
    transform = build_transform(architecture)
    return transform(image_pil.convert("RGB")).unsqueeze(0).to(DEVICE)


def as_display_array(image_pil, architecture):
    """The image resized to the model's input grid, as float RGB in [0, 1].

    This is the canvas the Grad-CAM heatmap is blended onto. Resizing the same
    way the model input is resized keeps the heatmap anatomically aligned.
    """
    size = PREPROCESSING[architecture]["input_size"]
    resized = image_pil.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def classify(model, input_tensor):
    """Return (predicted class name, confidence, index)."""
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        index = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, index].item())
    return CLASS_NAMES[index], confidence, index


def grad_cam(model, target_layer, input_tensor, class_index):
    """The raw (H, W) Grad-CAM saliency map for ``class_index``."""
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_index)]
    return cam(input_tensor=input_tensor, targets=targets)[0, :]


def blend_cam(display_image, grayscale_cam, opacity=0.5):
    """Draw a saliency map over ``display_image`` at the requested opacity."""
    return show_cam_on_image(
        display_image, grayscale_cam, use_rgb=True, image_weight=1.0 - opacity
    )
