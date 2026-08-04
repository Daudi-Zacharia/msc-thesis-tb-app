"""Model architectures and checkpoint loading."""

import torch
import torch.nn as nn

from config import (
    CHECKPOINT_FILENAMES,
    MODEL_URLS,
    STATE_DICT_CHECKPOINTS,
)
from download import ensure_checkpoint

DEVICE = torch.device("cpu")


class CustomCNN(nn.Module):
    """The hand-rolled baseline CNN, needed to rehydrate a bare state dict."""

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
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


def find_last_conv(model):
    """The last Conv2d in the graph — the Grad-CAM target layer."""
    last_conv = None
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found; cannot place a Grad-CAM hook.")
    return last_conv


def load_classifier(scheme, architecture, progress_hook=None):
    """Fetch (if needed) and load a classifier, returning (model, target_layer)."""
    url = MODEL_URLS[scheme][architecture]
    filename = CHECKPOINT_FILENAMES[scheme][architecture]
    path = ensure_checkpoint(url, filename, progress_hook=progress_hook)

    if (scheme, architecture) in STATE_DICT_CHECKPOINTS:
        model = CustomCNN()
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model = torch.load(path, map_location=DEVICE, weights_only=False)

    model = model.to(DEVICE)
    model.eval()
    # Grad-CAM backpropagates to the target layer, so gradients must flow.
    for param in model.parameters():
        param.requires_grad = True

    return model, find_last_conv(model)
