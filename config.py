"""Checkpoint locations and per-architecture preprocessing settings."""

CLASS_NAMES = ["Normal", "TB"]

ARCHITECTURES = [
    "Custom CNN",
    "DenseNet169",
    "VGG16",
    "ResNet50",
    "EfficientNetB3",
]

# Two checkpoint sets, same five architectures, different training schemes:
#   oversampling — trained on unsegmented D2 with oversampling; used by
#                  "Classify only", which feeds the raw upload to the model.
#   segmented    — trained on segmented D2; used by "Segment then classify",
#                  so the model sees the same kind of input it was trained on.
MODEL_URLS = {
    "oversampling": {
        "Custom CNN": "https://drive.google.com/file/d/1RuqyYxoriMQnynEzDix88kW5aKWUJlJ7/view?usp=sharing",
        "DenseNet169": "https://drive.google.com/file/d/12i2wPlarZMb_T1OjFOeXKknvLP3CygT7/view?usp=sharing",
        "VGG16": "https://drive.google.com/file/d/1H0l0DB3e8bGn7FOUc3a41vjD9HSh0cfX/view?usp=sharing",
        "ResNet50": "https://drive.google.com/file/d/1f-jG2999u7h3swXy1nKm_Zh9kscoejlG/view?usp=sharing",
        "EfficientNetB3": "https://drive.google.com/file/d/1s_NHkelNY1PouNKsGZvrMLAMtbXBAOrA/view?usp=sharing",
    },
    "segmented": {
        "Custom CNN": "https://drive.google.com/file/d/1aR8EfzUZxNSCAW7J88x1nVJrYRunGQrh/view?usp=sharing",
        "DenseNet169": "https://drive.google.com/file/d/1NVXQFkoTzGO8icunu3GE4_qYmrptRNh1/view?usp=drive_link",
        "VGG16": "https://drive.google.com/file/d/1w3ipjKaljm9vtvNI1R0dWTplOKzwuWug/view?usp=drive_link",
        "ResNet50": "https://drive.google.com/file/d/1mCS6mdoZRpiA2017FbFogtRNZYwim5hi/view?usp=sharing",
        "EfficientNetB3": "https://drive.google.com/file/d/1F8qISyMPXXeENjJo1lTjWdbmVgM9g9qf/view?usp=sharing",
    },
}

CHECKPOINT_FILENAMES = {
    "oversampling": {
        "Custom CNN": "class_ex3_run4_4200.pt",
        "DenseNet169": "densenet169_rahman_r5.pt",
        "VGG16": "vgg16_rahman_r3_2.pt",
        "ResNet50": "resnet50_rahman_r5.pt",
        "EfficientNetB3": "efficient_net_4200_run3.pt",
    },
    "segmented": {
        "Custom CNN": "class_seg_3cnn_4200.pt",
        "DenseNet169": "densenet169_class_seg_rahman.pt",
        "VGG16": "vgg16_class_seg_4200.pt",
        "ResNet50": "resnet50_class_seg_4200.pt",
        "EfficientNetB3": "enet_class_seg_4200.pt",
    },
}

# Only the oversampling Custom CNN was saved as a bare state dict; every other
# checkpoint (including the segmented Custom CNN) is a pickled model object.
STATE_DICT_CHECKPOINTS = {("oversampling", "Custom CNN")}

SEGMENTATION_MODEL_URL = (
    "https://drive.google.com/file/d/1Hc6GC45Gww0AFr_Ur6gh3VuVKrp0SJ3B/view?usp=sharing"
)
SEGMENTATION_CHECKPOINT = "seg_unet_2.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The Custom CNN was trained on plain [0, 1] tensors; the torchvision-backed
# architectures were trained with ImageNet normalisation.
PREPROCESSING = {
    "Custom CNN": {"input_size": 224, "mean": None, "std": None},
    "DenseNet169": {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "VGG16": {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "ResNet50": {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "EfficientNetB3": {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
}

SEGMENTATION_INPUT_SIZE = 256

MODES = {
    "Classify only": {"segment": False, "classify": True, "scheme": "oversampling"},
    "Segment only": {"segment": True, "classify": False, "scheme": None},
    "Segment then classify": {"segment": True, "classify": True, "scheme": "segmented"},
}

SCHEME_LABELS = {"oversampling": "oversampling", "segmented": "segmented"}
