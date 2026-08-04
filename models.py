"""Model architectures and checkpoint loading."""

import pickle

import torch
import torch.nn as nn
from torchvision import models as tv_models

from config import (
    CHECKPOINT_FILENAMES,
    MODEL_URLS,
    SEGMENTATION_CHECKPOINT,
    SEGMENTATION_MODEL_URL,
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


# The segmented Custom CNN was pickled from a training script where the class
# was named ``CNN`` in ``__main__``, so unpickling looks for ``__main__.CNN``.
# Its module layout is identical to CustomCNN (a single ``layers`` Sequential),
# and pickle restores the submodules themselves rather than re-running
# __init__, so resolving the name to CustomCNN reconstructs it exactly.
PICKLE_CLASS_ALIASES = {("__main__", "CNN"): CustomCNN}


class _AliasingUnpickler(pickle.Unpickler):
    """Resolves checkpoint classes that no longer live where they were saved.

    Injecting the name into ``__main__`` instead would not survive Streamlit,
    which builds a fresh ``__main__`` module on every script rerun while
    leaving this module's import-time state untouched.
    """

    def find_class(self, module, name):
        alias = PICKLE_CLASS_ALIASES.get((module, name))
        if alias is not None:
            return alias
        return super().find_class(module, name)


class _aliasing_pickle:
    """Duck-types the ``pickle`` module for ``torch.load(pickle_module=…)``."""

    Unpickler = _AliasingUnpickler
    load = pickle.load
    loads = pickle.loads


def _decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _up_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetVGG16(nn.Module):
    """U-Net with a VGG16-BN encoder, matching the ``seg_unet_2.pth`` layout.

    The encoder is sliced into five skip-producing blocks plus a bottleneck;
    the decoder upsamples back to the input resolution and emits two logit
    channels (background, lung).
    """

    def __init__(self, n_classes=2):
        super().__init__()
        self.encoder = tv_models.vgg16_bn(weights=None).features

        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])
        self.bottleneck = nn.Sequential(*self.encoder[34:])

        self.conv_bottleneck = _decoder_block(512, 1024)

        self.up_conv6 = _up_block(1024, 512)
        self.conv6 = _decoder_block(512 + 512, 512)
        self.up_conv7 = _up_block(512, 256)
        self.conv7 = _decoder_block(256 + 512, 256)
        self.up_conv8 = _up_block(256, 128)
        self.conv8 = _decoder_block(128 + 256, 128)
        self.up_conv9 = _up_block(128, 64)
        self.conv9 = _decoder_block(64 + 128, 64)
        self.up_conv10 = _up_block(64, 32)
        self.conv10 = _decoder_block(32 + 64, 32)

        self.conv11 = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.conv6(torch.cat([self.up_conv6(x), block5], dim=1))
        x = self.conv7(torch.cat([self.up_conv7(x), block4], dim=1))
        x = self.conv8(torch.cat([self.up_conv8(x), block3], dim=1))
        x = self.conv9(torch.cat([self.up_conv9(x), block2], dim=1))
        x = self.conv10(torch.cat([self.up_conv10(x), block1], dim=1))
        return self.conv11(x)


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
    """Fetch (if needed) and load a classifier, returning (model, target_layer).

    Two checkpoint sets exist per architecture. Most were saved with
    ``torch.save(model)``; the oversampling Custom CNN alone was saved as a
    state dict, so it is rebuilt from :class:`CustomCNN` first.
    """
    url = MODEL_URLS[scheme][architecture]
    filename = CHECKPOINT_FILENAMES[scheme][architecture]
    path = ensure_checkpoint(url, filename, progress_hook=progress_hook)

    if (scheme, architecture) in STATE_DICT_CHECKPOINTS:
        model = CustomCNN()
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model = torch.load(
            path, map_location=DEVICE, weights_only=False,
            pickle_module=_aliasing_pickle,
        )

    model = model.to(DEVICE)
    model.eval()
    # Grad-CAM backpropagates to the target layer, so gradients must flow.
    for param in model.parameters():
        param.requires_grad = True

    return model, find_last_conv(model)


def load_segmenter(progress_hook=None):
    """Fetch (if needed) and load the U-Net used by both segmentation modes."""
    path = ensure_checkpoint(
        SEGMENTATION_MODEL_URL, SEGMENTATION_CHECKPOINT, progress_hook=progress_hook
    )
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = UNetVGG16()
        model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()
    return model
