# Road Crack Detection — Pixel-Level Segmentation Using Deep Learning

A college deep-learning mini project that detects and segments pavement cracks at the pixel level using PyTorch. The system accepts road images, predicts a binary crack mask, and visualises results through a local Streamlit web application.

---

## Features

- Pixel-level crack segmentation (not just image-level classification)
- Two models: lightweight FCN baseline and U-Net with skip connections
- Combined BCE + Dice loss for handling severe class imbalance (~2 % crack pixels)
- Mixed-precision training (AMP) for RTX 2050 / limited VRAM GPUs
- Threshold tuned on validation set; test set never used for tuning
- Metrics: Dice, IoU, Precision, Recall (pixel accuracy reported but not ranked)
- Comprehensive dataset audit script
- 32-test pytest suite
- Streamlit application with dark inspection theme, adjustable threshold, overlay display

---

## Dataset

| Property | Value |
|---------|-------|
| Source | Roboflow `crack-seg` dataset |
| Total images | 4,028 (3,717 train / 200 val / 111 test) |
| Image size | 416 × 416 px (resized to 256 × 256 for training) |
| Mask encoding | 0 = background, 255 = crack |
| Mean crack coverage | ~2.2 % (severely imbalanced) |

Annotations were rasterised from YOLO-style polygon segments via `src/prepare_dataset.py`.

---

## Architectures

### FCN Baseline (~776 K parameters)
Encoder-decoder fully convolutional network **without** skip connections. Deliberately weaker than U-Net to provide a meaningful comparison baseline.

### U-Net (~7.8 M parameters)
Standard U-Net with 4-level encoder (32→64→128→256 channels), 512-channel bottleneck, and symmetric decoder with skip connections. Skip connections allow reconstruction of thin crack structures.

---

## Folder Structure

```
road-crack-detection/
├── app/
│   └── app.py               # Streamlit application
├── data/
│   └── raw/
│       ├── images/{train,val,test}/
│       └── masks/{train,val,test}/
├── models/
│   └── checkpoints/
│       ├── fcn_best.pth
│       └── unet_best.pth
├── outputs/
│   ├── audit/               # Dataset audit reports
│   ├── eval_fcn/            # FCN evaluation results
│   └── eval_unet/           # U-Net evaluation results
├── reports/
│   ├── literature_review.md
│   ├── experiment_plan.md
│   ├── evaluation_report.md
│   └── ...
├── src/
│   ├── config.py            # All hyperparameters and paths
│   ├── dataset.py           # PyTorch Dataset + DataLoader factory
│   ├── losses.py            # DiceLoss, CombinedLoss
│   ├── metrics.py           # Dice, IoU, Precision, Recall
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Test-set evaluation
│   ├── compare_models.py    # Side-by-side comparison
│   ├── audit_dataset.py     # Dataset validation
│   └── models/
│       ├── fcn_baseline.py
│       └── unet.py
├── tests/
│   └── test_pipeline.py     # 32 pytest unit tests
├── requirements.txt
└── README.md
```

---

## Environment Setup (Windows CMD)

> **Important:** All commands below assume you are in the project root and use the `.venv` virtual environment. Activate it once per terminal session.

```cmd
cd road-crack-detection
.venv\Scripts\activate
```

### Install dependencies

```cmd
pip install -r requirements.txt
```

---

## Commands

### Validate dataset

```cmd
python -m src.audit_dataset
```
Output saved to `outputs/audit/`.

### Train FCN baseline (30 epochs, ~41 min)

```cmd
python -m src.train --model fcn
```

### Train U-Net (50 epochs, ~70 min)

```cmd
python -m src.train --model unet
```

### Resume interrupted training

```cmd
python -m src.train --model unet --resume
```

### Evaluate FCN on test set

```cmd
python -m src.evaluate --model fcn
```

### Evaluate U-Net on test set

```cmd
python -m src.evaluate --model unet
```

### Compare both models

```cmd
python -m src.compare_models
```

### Launch Streamlit application

```cmd
.venv\Scripts\streamlit.exe run app\app.py
```

Then open your browser at `http://localhost:8501`.

### Run tests

```cmd
python -m pytest tests\ -v
```

---

## Training Configuration

All hyperparameters are in [`src/config.py`](src/config.py). Key defaults:

| Setting | FCN | U-Net |
|---------|-----|-------|
| Image size | 256 × 256 | 256 × 256 |
| Batch size | 8 | 8 |
| Epochs | 30 | 50 |
| Learning rate | 1e-3 | 1e-4 |
| Loss | 0.5×BCE + 0.5×Dice | 0.5×BCE + 0.5×Dice |
| Early stop | 10 epochs | 15 epochs |

---

## Reproducibility

- Fixed seed: `SEED = 42` in `src/config.py`
- `num_workers = 0` for Windows compatibility
- Deterministic DataLoader generator
- Checkpoints saved with full optimizer and scheduler state for exact resume

---

## Limitations

- VRAM constraint (4.29 GB RTX 2050) limits batch size and model depth
- No pretrained encoder (ResNet U-Net would improve results but exceeds VRAM)
- Single fixed threshold per model (per-image adaptive thresholding not implemented)
- No temporal/3D context — each image processed independently
- ~3,700 training images is moderate; model may struggle with unseen crack types

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `BATCH_SIZE` in `src/config.py` |
| `No module named src` | Run commands from the project root directory |
| `No checkpoint found` | Train the model first before evaluating |
| Streamlit port conflict | Add `--server.port 8502` to the streamlit command |
| Slow training on CPU | Install CUDA-enabled PyTorch; verify with `check-setup.py` |

---

## License

This project is for educational purposes. Dataset sourced from Roboflow.
