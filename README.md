# TB Chest X-ray Analyser

A Streamlit research prototype that segments the lung fields of a chest X-ray,
classifies it as **Normal** or **Tuberculosis**, and explains the prediction
with a Grad-CAM heatmap.

A live-hosted version can be accessed [here](https://msc-thesis-tb.streamlit.app/)

> **Research prototype — not a diagnostic tool. Do not use for clinical
> decision-making.**

Inference is single-image and CPU-only; no GPU is required.

## The three modes

| Mode | What happens | Checkpoint set |
|---|---|---|
| **Classify only** | The uploaded X-ray goes straight to the chosen classifier. Returns class, confidence and a Grad-CAM overlay. | oversampling |
| **Segment only** | A U-Net predicts the lung mask; the masked image is shown beside the original and can be downloaded. | — (U-Net) |
| **Segment then classify** | The U-Net segments first, and the *segmented* image is fed to the chosen classifier. | segmented |

You pick the mode and the architecture in the sidebar. The same five
architecture names are offered in both classification modes — the app resolves
which checkpoint to load from the `(mode, architecture)` pair.

## Two classification checkpoint sets

There are two checkpoints per architecture, trained under different schemes,
and each mode uses the one that matches its input:

- **Oversampling set** — trained on **unsegmented** D2 with oversampling to
  address class imbalance. Used by *Classify only*, which feeds the raw upload
  to the model.
- **Segmented set** — trained on **segmented** D2. Used by *Segment then
  classify*, so the model is evaluated on the same kind of input it was
  trained on.

Keeping the sets separate is the point: running an unsegmented-trained model
on a segmented image (or vice versa) is a train/inference mismatch and its
confidences would not mean much.

| Architecture | Oversampling checkpoint | Segmented checkpoint |
|---|---|---|
| Custom CNN | `class_ex3_run4_4200.pt` | `class_seg_3cnn_4200.pt` |
| DenseNet169 | `densenet169_rahman_r5.pt` | `densenet169_class_seg_rahman.pt` |
| VGG16 | `vgg16_rahman_r3_2.pt` | `vgg16_class_seg_4200.pt` |
| ResNet50 | `resnet50_rahman_r5.pt` | `resnet50_class_seg_4200.pt` |
| EfficientNetB3 | `efficient_net_4200_run3.pt` | `enet_class_seg_4200.pt` |

Segmentation uses `seg_unet_2.pth`, a U-Net with a VGG16-BN encoder.

## Grad-CAM

The heatmap is computed from whatever tensor the model actually receives — the
segmented image in *Segment then classify* — but it is always rendered over
the **original** X-ray. Overlaying on a mostly-black masked image is hard to
read; overlaying on the full X-ray keeps the heatmap in anatomical context.
The sidebar has an opacity slider (0–100%); moving it re-blends the cached
saliency map rather than re-running inference.

## Model weights

Weights are **not** in this repository. On first use of a given
mode/architecture the app downloads that checkpoint from Google Drive with
`gdown` into `models/` and reuses it from there on every later run. Only the
checkpoints needed for the current selection are fetched — never all eleven up
front. Within a session, loaded models are also cached in `st.session_state`
keyed by `(scheme, architecture)`, so switching back to a model you have
already used is instant.

Drive locations live in [`config.py`](config.py); `download.py` accepts either
a full "anyone with the link" share URL or a bare file ID.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sample chest X-rays live under `sample_images/tb/` and `sample_images/normal/`
and can be loaded from the sidebar's "…or try an example" selector without
uploading anything.

## Layout

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI and orchestration |
| `config.py` | Checkpoint URLs/filenames, class names, preprocessing settings |
| `download.py` | Google Drive fetch and on-disk caching |
| `models.py` | Architectures (`CustomCNN`, `UNetVGG16`) and checkpoint loading |
| `inference.py` | Preprocessing, prediction, segmentation, Grad-CAM |
