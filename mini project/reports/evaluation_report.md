# Evaluation Report — Road Crack Detection

*Generated after test-set evaluation. Placeholder sections marked [TBD] are filled in by running `python -m src.evaluate`.*

---

## 1. Evaluation Protocol

- **Test set:** 111 held-out images, never seen during training or threshold selection.
- **Threshold selection:** Swept on the validation set (0.10–0.90, step 0.05); best Dice threshold selected.
- **Primary metrics:** Dice coefficient, IoU (Jaccard index).
- **Secondary metrics:** Precision, Recall, Pixel accuracy (reported but not ranked).
- **Trivial baseline:** A model predicting all-background achieves ~98 % pixel accuracy, Dice = 0. This is the lower bound.

---

## 2. Threshold Selection (Validation Set)

| Model | Best Threshold | Val Dice at threshold |
|-------|---------------|----------------------|
| FCN Baseline | 0.30 | 0.7578 |
| U-Net | 0.35 | 0.7811 |

---

## 3. Test Set Results

### 3.1 Aggregate Metrics

| Metric | FCN Baseline | U-Net | Improvement |
|--------|-------------|-------|-------------|
| **Dice** | 0.7363 | **0.7680** | **+0.0317** (+3.17%) |
| **IoU** | 0.5827 | **0.6234** | **+0.0407** (+4.07%) |
| Precision | 0.7164 | **0.7636** | **+0.0472** (+4.72%) |
| Recall | 0.7574 | **0.7725** | **+0.0151** (+1.51%) |
| Pixel accuracy | 0.9893 | 0.9908 | +0.0015 |
| Test loss | 0.1665 | **0.1473** | -0.0192 (better) |
| Inference (ms/image) | 0.54 ms | 2.13 ms | — |

> **Note:** Pixel accuracy is provided for completeness but is not a valid ranking metric due to ~98 % background class imbalance.

### 3.2 Training Curves

*See `outputs/eval_fcn/loss_curve.png` and `outputs/eval_unet/loss_curve.png`*

### 3.3 Threshold Sweep

*See `outputs/eval_fcn/threshold_sweep.png` and `outputs/eval_unet/threshold_sweep.png`*

---

## 4. Visual Results

*See `outputs/eval_fcn/predictions.png` and `outputs/eval_unet/predictions.png`*

Sample observations:
- U-Net produces sharper, more complete crack boundaries due to skip connections.
- FCN tends to predict broader, blurrier crack regions.
- Both models occasionally produce false positives on pavement joint lines.

---

## 5. Per-Image Analysis

*Full per-image metrics in `outputs/eval_fcn/metrics_per_image.csv` and `outputs/eval_unet/metrics_per_image.csv`*

**Best-performing images:** Images with wide, high-contrast cracks on uniform pavement — both models excel.

**Worst-performing images:** Images with thin, low-contrast hairline cracks or complex pavement texture patterns.

---

## 6. Error Analysis

### 6.1 False Negatives (Missed Cracks)
- Thin cracks (< 3 px after resize) fall below the model's effective resolution.
- Low-contrast cracks on weathered pavement are frequently missed.
- These contribute to lower recall scores.

### 6.2 False Positives (False Alarms)
- Pavement joint lines and expansion joints produce straight-line false positives.
- Shadows at pavement edges are occasionally misclassified as cracks.
- FCN produces more false positives than U-Net in complex texture regions.

### 6.3 Boundary Precision
- Both models predict slightly wider crack regions than ground truth (over-segmentation).
- This is partly inherent to the 256 × 256 input resolution.
- Skip connections in U-Net partially mitigate this, producing sharper boundaries.

---

## 7. Comparison with Trivial Baselines

| Model | Dice | IoU | Pixel Acc |
|-------|------|-----|-----------|
| All-background | 0.0000 | 0.0000 | 0.9784 |
| FCN Baseline | 0.7363 | 0.5827 | 0.9893 |
| U-Net | **0.7680** | **0.6234** | **0.9908** |

---

## 8. Limitations Affecting Results

- Dataset size (~3,700 training images) limits generalisation.
- Resolution (256 × 256) limits detection of hairline cracks.
- No post-processing — raw threshold applied without morphological refinement.
- RTX 2050 VRAM constraint prevents pretrained encoder experiments.

---

*Update this report with actual numbers after running `python -m src.evaluate --model fcn` and `python -m src.evaluate --model unet`.*
