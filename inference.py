"""Preprocessing, prediction, segmentation and Grad-CAM."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms as T

from config import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSING,
    SEGMENTATION_INPUT_SIZE,
)

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


# -------------------- classification --------------------

def classify(model, input_tensor):
    """Return (predicted class name, confidence, index)."""
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        index = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, index].item())
    return CLASS_NAMES[index], confidence, index


def grad_cam(model, target_layer, input_tensor, class_index):
    """The raw (H, W) Grad-CAM saliency map for ``class_index``.

    Kept separate from blending so the UI can re-render at a new opacity
    without paying for another backward pass.
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_index)]
    return cam(input_tensor=input_tensor, targets=targets)[0, :]


def blend_cam(display_image, grayscale_cam, opacity=0.5):
    """Draw a saliency map over ``display_image`` at the requested opacity.

    In "Segment then classify" the saliency map comes from the segmented
    image (what the model actually sees) while ``display_image`` is the
    original X-ray — a mostly-black masked image makes for an unreadable
    overlay.
    """
    return show_cam_on_image(
        display_image, grayscale_cam, use_rgb=True, image_weight=1.0 - opacity
    )


# -------------------- segmentation --------------------

def _mask_from_logits(logits):
    """Collapse a U-Net output of either shape into a single (H, W) mask."""
    if logits.shape[1] == 1:
        probability = torch.sigmoid(logits)[0, 0]
    else:
        probability = torch.softmax(logits, dim=1)[0, 1]
    return (probability > 0.5).float()


SEGMENTATION_TRANSFORM = T.Compose([
    T.Resize((SEGMENTATION_INPUT_SIZE, SEGMENTATION_INPUT_SIZE)),
    T.ToTensor(),
    # The U-Net's encoder is a VGG16-BN backbone; it was trained on
    # ImageNet-normalised input and predicts near-empty masks without this.
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def segment(unet, image_pil):
    """Run the U-Net and return (masked RGB image, binary mask as uint8).

    The mask is predicted at the U-Net's own resolution and then resized back
    to the original image size, so the returned image keeps its input shape.
    """
    original = image_pil.convert("RGB")
    width, height = original.size
    tensor = SEGMENTATION_TRANSFORM(original).unsqueeze(0)

    with torch.no_grad():
        logits = unet(tensor.to(DEVICE))
    mask = _mask_from_logits(logits)

    mask_full = F.interpolate(
        mask[None, None], size=(height, width), mode="nearest"
    )[0, 0].numpy()

    array = np.asarray(original, dtype=np.uint8)
    masked = (array * mask_full[..., None]).astype(np.uint8)
    return Image.fromarray(masked), (mask_full * 255).astype(np.uint8)
