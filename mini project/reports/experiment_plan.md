# Experiment Plan — Road Crack Detection and Pixel-Level Segmentation

## 1. Research Question

Can a U-Net segmentation model trained on a moderate-sized pavement crack dataset (~3,700 images) produce reliable pixel-level crack masks suitable for infrastructure inspection support, and does it measurably outperform a simpler FCN baseline on Dice coefficient and IoU?

---

## 2. Problem Definition

**Task:** Binary semantic segmentation — assign each pixel in a road image one of two labels: crack (foreground) or background.

**Input:** RGB road/pavement image, 256 × 256 px (resized from 416 × 416 originals).

**Output:** Single-channel probability map in [0, 1]; thresholded to a binary mask for evaluation.

**Challenge:** Extreme class imbalance (~2 % foreground). Cracks are thin, low-contrast, and irregular. Standard pixel accuracy is an inadequate metric.

---

## 3. Dataset Description

| Property | Value |
|---------|-------|
| Source | Roboflow crack-seg dataset (NDJSON manifest) |
| Original size | 416 × 416 px |
| Model input size | 256 × 256 px |
| Train images | 3,717 |
| Val images | 200 |
| Test images | 111 |
| Mask encoding | 0 = background, 255 = crack |
| Mean crack coverage (train) | 2.165 % |
| Fully-empty masks | 0 (train), 1 (val), 0 (test) |
| Cross-split duplicates | None detected |

Annotation method: polygon segments from NDJSON manifest, rasterised to binary masks via `cv2.fillPoly`.

---

## 4. Baseline Architecture — Small FCN Encoder-Decoder

A fully convolutional encoder-decoder **without** skip connections. The encoder uses strided convolutions (or max pooling) to reduce spatial resolution; the decoder uses bilinear upsampling followed by convolutions to restore it.

```
Input (3, 256, 256)
  -> Encoder Block 1: Conv3x3(3->32) + BN + ReLU + MaxPool2x2  -> (32, 128, 128)
  -> Encoder Block 2: Conv3x3(32->64) + BN + ReLU + MaxPool2x2 -> (64, 64, 64)
  -> Encoder Block 3: Conv3x3(64->128) + BN + ReLU + MaxPool2x2 -> (128, 32, 32)
  -> Bottleneck:      Conv3x3(128->256) + BN + ReLU             -> (256, 32, 32)
  -> Decoder Block 1: Upsample 2x + Conv3x3(256->128) + BN + ReLU -> (128, 64, 64)
  -> Decoder Block 2: Upsample 2x + Conv3x3(128->64) + BN + ReLU  -> (64, 128, 128)
  -> Decoder Block 3: Upsample 2x + Conv3x3(64->32) + BN + ReLU   -> (32, 256, 256)
  -> Head:            Conv1x1(32->1)                              -> (1, 256, 256)
Output: logit map (sigmoid for inference)
```

Parameters: ~1.3 M. Deliberate weakness: no skip connections, so fine crack detail is lost during downsampling.

---

## 5. Main Architecture — U-Net

Standard U-Net with a symmetric encoder-decoder and skip connections. Concatenating encoder feature maps to decoder layers allows the model to recover thin spatial detail.

```
Input (3, 256, 256)
Encoder:
  E1: 2x Conv3x3(3->32) + BN + ReLU                -> (32, 256, 256); skip S1
  MaxPool2x2                                         -> (32, 128, 128)
  E2: 2x Conv3x3(32->64) + BN + ReLU               -> (64, 128, 128); skip S2
  MaxPool2x2                                         -> (64, 64, 64)
  E3: 2x Conv3x3(64->128) + BN + ReLU              -> (128, 64, 64); skip S3
  MaxPool2x2                                         -> (128, 32, 32)
  E4: 2x Conv3x3(128->256) + BN + ReLU             -> (256, 32, 32); skip S4
  MaxPool2x2                                         -> (256, 16, 16)
Bottleneck:
  2x Conv3x3(256->512) + BN + ReLU                 -> (512, 16, 16)
Decoder:
  D4: TransposedConv(512->256) + Cat(S4) + 2xConv(512->256)  -> (256, 32, 32)
  D3: TransposedConv(256->128) + Cat(S3) + 2xConv(256->128)  -> (128, 64, 64)
  D2: TransposedConv(128->64)  + Cat(S2) + 2xConv(128->64)   -> (64, 128, 128)
  D1: TransposedConv(64->32)   + Cat(S1) + 2xConv(64->32)    -> (32, 256, 256)
Head: Conv1x1(32->1)                                          -> (1, 256, 256)
Output: logit map (sigmoid for inference)
```

Parameters: ~7.8 M. Skip connections allow the decoder to reconstruct thin cracks that the encoder would otherwise smear.

---

## 6. Preprocessing and Augmentation

### 6.1 Common Preprocessing (all splits)
- Resize image and mask to 256 × 256 (bilinear for image, nearest-neighbour for mask).
- Normalise image: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (ImageNet statistics).
- Convert mask to binary tensor: values 0 and 1.

### 6.2 Training Augmentations (train split only)
Augmentations are applied **jointly** to image and mask (same spatial transform for both).

| Augmentation | Settings | Rationale |
|-------------|---------|-----------|
| Horizontal flip | p=0.5 | Crack orientation is arbitrary |
| Vertical flip | p=0.5 | Crack orientation is arbitrary |
| Random rotation | ±15°, p=0.5 | Handles angled cracks |
| Colour jitter (brightness, contrast) | 0.2 factor, p=0.5 | Handles varying lighting; applied to image only |

**Excluded augmentations:**
- Colour inversion (destroys crack texture distinguishability).
- Elastic distortion (could artificially merge or break thin cracks).
- Aggressive crops (risk cutting annotated cracks out of frame).

---

## 7. Hyperparameters

| Parameter | FCN Baseline | U-Net | Rationale |
|-----------|-------------|-------|-----------|
| Image size | 256 × 256 | 256 × 256 | Fits RTX 2050 (4.29 GB VRAM) |
| Batch size | 8 | 8 | VRAM budget |
| Epochs | 30 | 50 | U-Net needs longer to converge |
| Optimizer | Adam | Adam | Adaptive LR; robust default |
| Learning rate | 1e-3 | 1e-4 | U-Net is deeper; lower LR is safer |
| LR scheduler | ReduceLROnPlateau (patience=5) | ReduceLROnPlateau (patience=5) | Adapt to plateau |
| Loss function | 0.5*BCE + 0.5*Dice | 0.5*BCE + 0.5*Dice | Handles class imbalance |
| Early stopping | patience=10 | patience=15 | Prevent overfitting |
| Seed | 42 | 42 | Reproducibility |
| num_workers | 0 | 0 | Windows multiprocessing safety |

---

## 8. Evaluation Metrics

All reported on the **held-out test set only**, after model selection on validation loss.

| Metric | Primary use |
|--------|------------|
| Dice coefficient | Primary ranking metric |
| IoU (Jaccard) | Secondary ranking metric |
| Precision | False-positive characterisation |
| Recall | False-negative characterisation |
| Pixel confusion matrix | Visualise TP/FP/TN/FN |
| Pixel accuracy | Reported but not used for ranking |

**Threshold selection:** A threshold is swept on the validation set (0.1 to 0.9 in steps of 0.05). The threshold maximising validation Dice is applied to test predictions. The test set is never used for threshold tuning.

---

## 9. Experiment Comparison Strategy

| Step | What we compare |
|------|----------------|
| Baseline training | FCN vs. all-background trivial baseline |
| Main model training | U-Net vs. FCN baseline |
| Visual inspection | Qualitative comparison of predicted masks |
| Error analysis | Categories: missed thin cracks, false positives on texture, boundary precision |

All runs use identical seeds, data splits, and preprocessing to ensure fair comparison.

---

## 10. Expected Limitations

- **VRAM constraint:** 4.29 GB limits batch size and channel depth. A ResNet-backed U-Net would likely improve results but exceeds hardware.
- **Dataset size:** ~3,700 training images is moderate. The model may struggle with rarely seen crack types.
- **Single threshold:** Per-image adaptive thresholding could improve results but complicates the pipeline.
- **No post-processing:** Morphological operations (skeleton extraction, connected components) could sharpen crack boundaries but are not implemented in this project scope.
- **2D only:** The model operates on individual frames with no temporal or 3D context.

---

## 11. Files to be Created

| File | Purpose |
|------|---------|
| `src/config.py` | All hyperparameters and paths |
| `src/dataset.py` | PyTorch Dataset and DataLoader factory |
| `src/models/fcn_baseline.py` | FCN encoder-decoder baseline |
| `src/models/unet.py` | U-Net main model |
| `src/losses.py` | Combined BCE+Dice loss |
| `src/metrics.py` | Dice, IoU, Precision, Recall |
| `src/train.py` | Training loop with checkpointing |
| `src/evaluate.py` | Test-set evaluation and plots |
| `app/app.py` | Streamlit application |
| `tests/` | Pytest unit tests |

---

*Plan drafted: 2026-10-21. All architecture details and hyperparameters are proposals; actual performance is not yet known.*
