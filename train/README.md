# Training & Conversion Pipeline

The machine-learning pipeline behind Arcane Gambit's dice recognition. It trains
YOLO (Ultralytics) models, runs a two-stage detect-then-classify inference flow,
and exports the trained weights to ONNX for consumption by the native
[`cv/`](../cv/) library.

## Approach

Dice recognition is split into two models:

1. **Detection** (`runs/dice_detect.*`) — locates dice in the frame.
2. **Classification** (`runs/dice_classify_2.*`) — reads the face value of each
   cropped die. `hybrid_detection.py` chains the two and maps raw class IDs to
   die values (1–20).

Training on the cropped, single-die classification task proved far more accurate
than asking one detector to both locate and read faces directly.

## Requirements

- Python 3.8+
- [Ultralytics YOLO](https://docs.ultralytics.com/), PyTorch, OpenCV, ONNX

```sh
pip install ultralytics torch opencv-python onnx
```

## Layout

```
train/
├── trainers/          # Model training entry points
│   ├── detect.py        - train the dice detection model
│   └── classify.py      - train the face-value classifier
├── helpers/           # Dataset preparation utilities
│   ├── yoloify.py       - convert annotations to YOLO format
│   ├── create_val.py    - build a validation split
│   ├── image_crop.py    - crop dice for the classification dataset
│   ├── update_labels.py / class_zeroer.py / image_rename.py / delete_cropped.py
│   └── export.py        - export a trained .pt model to ONNX
├── runs/              # Trained weights (.pt) and exported models (.onnx)
├── detect_camera.py   # Live detection from a webcam
├── detect_image.py    # Detection on a single image
├── classify_camera.py # Live classification
├── classify_image.py  # Classification on a single image
├── detect_crop.py     # Detect and crop dice
├── crop_label.py      # Crop + label helper
└── hybrid_detection.py# Full detect -> classify -> value-mapping flow
```

## Usage

```sh
# Train
python trainers/detect.py
python trainers/classify.py

# Export a trained model to ONNX (consumed by the cv/ DLL)
python helpers/export.py

# Run the full hybrid pipeline on a webcam
python hybrid_detection.py
```

The exported `runs/*.onnx` models are loaded at runtime by the C++ library in
[`cv/`](../cv/) via its `InitializeNetworks` entry point.
