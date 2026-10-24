# Presentation Outline — Road Crack Detection Using Deep Learning

*Duration: 10–15 minutes (adjust slide count to your time slot)*

---

## Slide 1 — Title

**Road Crack Detection and Pixel-Level Segmentation Using Deep Learning**

- Your name, roll number, institution
- Mini project submitted to [Department] — [Semester, Year]
- Guide: [Supervisor name]

---

## Slide 2 — Motivation

- Road network is critical infrastructure; cracks are early damage indicators
- Manual inspection: slow, expensive, subjective, dangerous
- Deep learning enables automated, consistent, scalable detection
- Show one striking before/after image: cracked road → model overlay

**Talking point:** "Traditional methods rely on engineers walking roads with clipboards. We can automate this with a camera and a trained model."

---

## Slide 3 — Problem Statement

- **Task:** Binary semantic segmentation — label every pixel as crack or background
- **Not** image classification (no pixel-level output)
- **Challenge:** Cracks occupy only ~2 % of image area — severe class imbalance
- Show: example image | ground-truth mask | what a bad (all-background) prediction looks like

---

## Slide 4 — Dataset

| Property | Value |
|---------|-------|
| Source | Roboflow crack-seg |
| Total images | 4,028 |
| Train / Val / Test | 3,717 / 200 / 111 |
| Image size | 416 × 416 → resized to 256 × 256 |
| Mean crack coverage | 2.2 % |

- Annotated as polygons → rasterised to binary masks
- Verified: no corrupt files, no cross-split leakage

---

## Slide 5 — Methodology Overview

Flowchart:
```
Raw images + polygon masks
      ↓
Preprocessing (resize, normalise)
      ↓
Training augmentation (flip, rotate, jitter)
      ↓
Model (FCN or U-Net)
      ↓
Combined BCE + Dice loss
      ↓
Checkpoint selection (best val loss)
      ↓
Threshold tuning (val set)
      ↓
Test-set evaluation
```

---

## Slide 6 — Baseline: FCN Encoder-Decoder

- 3 encoder blocks (Conv + BN + ReLU + MaxPool)
- Bottleneck (256 channels)
- 3 decoder blocks (Bilinear upsample + Conv)
- 1 × 1 head → single logit channel
- **No skip connections** — deliberate limitation
- ~776 K parameters

Diagram: show the encoder-decoder "V" shape without skip connections.

---

## Slide 7 — Main Model: U-Net

- 4-level encoder: 32→64→128→256 channels
- 512-channel bottleneck
- 4-level decoder with **skip connections** (concatenation)
- ~7.8 M parameters

Diagram: classic U-Net "U" shape with skip connection arrows.

**Key point:** Skip connections let the decoder recover thin crack boundaries that pooling would otherwise erase.

---

## Slide 8 — Loss Function

- Standard cross-entropy is dominated by background (98 %) → model learns to predict all-background
- **Dice loss** = 1 − 2·TP / (2·TP + FP + FN) — directly optimises overlap
- **Combined:** 0.5 × BCE + 0.5 × Dice
  - BCE → per-pixel probability calibration
  - Dice → structure-level recall

Show: diagram or equation.

---

## Slide 9 — Training Setup

| Setting | FCN | U-Net |
|---------|-----|-------|
| Epochs | 30 | 50 |
| Batch | 8 | 8 |
| LR | 1e-3 | 1e-4 |
| GPU | RTX 2050 (4.29 GB) | same |
| AMP | Yes | Yes |
| Early stop | 10 epochs | 15 epochs |

- ReduceLROnPlateau scheduler
- Best checkpoint saved by validation loss

---

## Slide 10 — Evaluation Metrics

| Metric | Formula | Why we use it |
|--------|---------|--------------|
| Dice | 2·TP / (2·TP+FP+FN) | Primary — handles imbalance |
| IoU | TP / (TP+FP+FN) | Secondary — stricter |
| Precision | TP / (TP+FP) | False alarm rate |
| Recall | TP / (TP+FN) | Crack detection rate |
| Pixel Acc | (TP+TN) / Total | Reported but **not ranked** (misleading) |

**Key point:** A model predicting all-background scores 98 % pixel accuracy but 0 Dice.

---

## Slide 11 — Results

| Metric | FCN Baseline | U-Net | Improvement |
|--------|-------------|-------|-------------|
| **Dice** | 0.7363 | **0.7680** | **+0.0317** (+3.2%) |
| **IoU** | 0.5827 | **0.6234** | **+0.0407** (+4.1%) |
| **Precision** | 0.7164 | **0.7636** | **+0.0472** (+4.7%) |
| **Recall** | 0.7574 | **0.7725** | **+0.0151** (+1.5%) |
| Inference (ms) | 0.54 ms | 2.13 ms | Real-time |

- Val-tuned thresholds: FCN ($t = 0.30$), U-Net ($t = 0.35$)
- Both models dramatically beat all-background trivial baseline ($Dice = 0.0$)
- U-Net outperforms FCN across **every metric** due to skip connections preserving fine crack boundaries

---

## Slide 12 — Visual Results

Show 4–6 test images in a grid:
- Column 1: Original image
- Column 2: Ground-truth mask
- Column 3: FCN prediction
- Column 4: U-Net prediction

Point out: U-Net produces sharper, more complete crack regions. FCN misses thin cracks.

---

## Slide 13 — Error Analysis

**Failure cases:**
- Thin, low-contrast cracks → missed (false negatives)
- Pavement texture / joint lines → false positives
- Boundary imprecision → predictions slightly wider/shorter than ground truth

**What this means:** The model is usable as a screening tool but not for engineering measurement. Human review of flagged areas is still essential.

---

## Slide 14 — Application Demo

Screenshot of the Streamlit app showing:
- Upload panel
- Original image | predicted mask | overlay
- Threshold slider
- Crack coverage %, inference time
- Disclaimer

Launch command: `.venv\Scripts\streamlit.exe run app\app.py`

---

## Slide 15 — Conclusions and Future Work

**Conclusions:**
- U-Net with combined BCE+Dice loss achieves meaningful pixel-level crack detection
- Skip connections are critical for reconstructing thin crack structures
- Dice/IoU are the correct primary metrics for imbalanced segmentation

**Future improvements:**
- Pretrained ResNet encoder (transfer learning)
- Test-time augmentation for more stable predictions
- Larger, more diverse dataset
- Post-processing (morphological thinning)

---

## Slide 16 — References

1. Ronneberger et al. (2015) — U-Net. MICCAI.
2. Long et al. (2015) — FCN. CVPR.
3. Zou et al. (2019) — DeepCrack. IEEE TIP.
4. Milletari et al. (2016) — V-Net / Dice loss. 3DV.
5. Yang et al. (2019) — Crack500 dataset. IEEE TITS.

---

## Slide 17 — Thank You / Q&A

"Thank you. I'm happy to answer questions."

**Prepare for:**
- Why Dice over accuracy?
- How does U-Net differ from FCN?
- What would you do with more time/resources?
- Can this be deployed on a road-inspection drone?
