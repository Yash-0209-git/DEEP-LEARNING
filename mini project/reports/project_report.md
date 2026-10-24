# Project Report — Road Crack Detection and Pixel-Level Segmentation Using Deep Learning

*College Mini Project Report | Deep Learning | [Academic Year]*

---

## Abstract

Road surface cracks are an early indicator of pavement deterioration and pose safety risks if left undetected. This project presents an end-to-end system for automated pixel-level crack detection in road images using convolutional neural networks. We train and evaluate two models on a dataset of 4,028 annotated road images: a small fully convolutional baseline (FCN) and a U-Net with skip connections. Both models are trained with a combined Binary Cross-Entropy and Dice loss to address the severe class imbalance inherent in crack segmentation (~2 % foreground pixels). Evaluation on a held-out test set uses Dice coefficient and Intersection over Union as primary metrics. A local Streamlit web application provides interactive crack detection. The project demonstrates that U-Net's skip connections produce measurably sharper segmentation than a plain encoder-decoder, and establishes a reproducible experimental framework for future work.

---

## 1. Introduction

Road infrastructure is critical to economic activity and public safety. Pavement deterioration typically begins with surface cracking caused by traffic loads, thermal cycles, and material fatigue. Early detection and quantification of cracks enables timely maintenance, preventing costlier structural repairs.

Traditional crack inspection relies on trained engineers performing visual surveys, which is slow, subjective, and expensive at scale. Camera-equipped vehicles can capture pavement imagery efficiently, but automated analysis of these images remains challenging due to low crack contrast, complex pavement texture, variable lighting, and the extreme spatial sparsity of crack regions.

Deep learning has transformed computer vision since 2012. Convolutional neural networks have been applied to pavement crack detection since approximately 2016, with pixel-level segmentation methods outperforming earlier patch-based classifiers. This project implements, trains, and evaluates two segmentation architectures on a publicly available annotated dataset, assesses their performance quantitatively and qualitatively, and packages the best model in an interactive application.

---

## 2. Problem Statement

Given a colour road/pavement image, produce a binary segmentation mask that labels each pixel as either **crack** (foreground) or **background**. The mask must have the same spatial dimensions as the input image.

This is distinct from:
- **Image classification** (binary label per image, no spatial output)
- **Object detection** (bounding boxes around crack regions, not pixel-precise)

Pixel-level segmentation provides the crack location, shape, length, and approximate area — all necessary for engineering assessment.

---

## 3. Objectives

1. Validate and characterise the training dataset.
2. Implement a reproducible data preprocessing and augmentation pipeline.
3. Train a small FCN encoder-decoder as a quantitative baseline.
4. Train a U-Net model as the primary segmentation system.
5. Evaluate both models on Dice coefficient, IoU, Precision, and Recall.
6. Analyse failure cases and model limitations.
7. Build an interactive Streamlit application for local inference.
8. Document the project for reproducibility.

---

## 4. Related Work

### 4.1 Fully Convolutional Networks
Long et al. (2015) demonstrated that classification CNNs can be adapted for dense prediction by replacing fully connected layers with convolutions. Their FCN-8s uses skip connections from shallow layers to improve spatial resolution. This is the direct precursor to our FCN baseline (without skip connections) and U-Net (with skip connections).

### 4.2 U-Net
Ronneberger et al. (2015) introduced U-Net for biomedical image segmentation. Its symmetric encoder-decoder with skip connections allows precise boundary localisation with limited annotated data — characteristics shared with crack segmentation tasks. U-Net is now the dominant architecture for thin-structure segmentation.

### 4.3 Crack-Specific Methods
Zhang et al. (2016) applied CNNs to patch-level crack detection. Zou et al. (2019) proposed DeepCrack, aggregating multi-scale features from a VGG backbone for pixel-level prediction. Yang et al. (2019) introduced the Crack500 dataset and a feature-pyramid network, reporting F-measure of 0.57 on a challenging benchmark.

### 4.4 Loss for Imbalanced Segmentation
Milletari et al. (2016) introduced Dice loss for volumetric medical segmentation. Because Dice loss is defined over the foreground region rather than all pixels, it is naturally robust to class imbalance. We adopt a combined BCE + Dice loss following established practice in crack and medical segmentation.

---

## 5. Dataset Description

The dataset was sourced from Roboflow as an NDJSON manifest (`crack-seg.ndjson`). Images are 416 × 416 px colour photographs of road surfaces annotated with polygon-based crack regions.

| Split | Images | Masks | Mean crack % |
|-------|--------|-------|-------------|
| Train | 3,717 | 3,717 | 2.165 % |
| Val | 200 | 200 | 2.026 % |
| Test | 111 | 111 | 1.977 % |

Masks were rasterised from YOLO-style normalised polygon coordinates using `cv2.fillPoly`, producing binary images (0 = background, 255 = crack). An automated audit verified: zero corrupt files, zero missing mask pairs, zero wrong dimensions (all 416 × 416), zero mask values outside {0, 255}, and zero cross-split filename duplicates.

**Class imbalance:** Crack pixels constitute ~2 % of the image. A model predicting all-background achieves ~98 % pixel accuracy. This makes pixel accuracy an invalid primary metric for this task.

---

## 6. Proposed Methodology

### 6.1 Preprocessing
- Resize image and mask to 256 × 256 (bilinear for image, nearest-neighbour for mask)
- Normalise image with ImageNet statistics: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- Convert mask to binary float tensor {0.0, 1.0}

### 6.2 Augmentation (training only, applied identically to image and mask)
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- Random rotation ±15° (p=0.5)
- Colour jitter — brightness/contrast ±20 % (p=0.5, image only)

### 6.3 Loss Function
Combined loss = 0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss

$$L = 0.5 \cdot \text{BCE}(y_{logit}, y) + 0.5 \cdot \left(1 - \frac{2\sum p_i t_i + \epsilon}{\sum p_i + \sum t_i + \epsilon}\right)$$

### 6.4 Optimiser and Scheduler
- Adam optimiser
- ReduceLROnPlateau (factor 0.5, patience 5 epochs)
- Early stopping (patience 10/15 epochs for FCN/U-Net)

### 6.5 Threshold Selection
Threshold swept on the validation set (0.10–0.90, step 0.05). Best Dice threshold applied to all test predictions. Test set is never used for threshold selection.

---

## 7. Model Architectures

### 7.1 FCN Baseline (~776 K parameters)

```
Input [3, 256, 256]
  Encoder: 3 × (Conv3x3 + BN + ReLU + MaxPool2x2)
    Channels: 3 → 32 → 64 → 128
    Spatial:  256 → 128 → 64 → 32
  Bottleneck: Conv3x3(128→256) + BN + ReLU
  Decoder: 3 × (Bilinear 2x + Conv3x3 + BN + ReLU)
    Channels: 256 → 128 → 64 → 32
    Spatial:  32 → 64 → 128 → 256
  Head: Conv1x1(32→1)
Output [1, 256, 256] logits
```

**No skip connections.** Spatial detail lost during encoding cannot be recovered during decoding.

### 7.2 U-Net (~7.8 M parameters)

```
Input [3, 256, 256]
  Encoder: 4 × (2×Conv3x3 + BN + ReLU + MaxPool2x2) → skip maps S1–S4
    Channels: 3→32→64→128→256
  Bottleneck: 2×Conv3x3(256→512) + BN + ReLU
  Decoder: 4 × (TransposedConv + Cat(skip) + 2×Conv3x3 + BN + ReLU)
    Channels: 512→256→128→64→32
  Head: Conv1x1(32→1)
Output [1, 256, 256] logits
```

Skip connections concatenate encoder feature maps to decoder layers, providing fine spatial detail for thin crack reconstruction.

---

## 8. Implementation Details

| Component | Library / Tool |
|-----------|---------------|
| Deep learning | PyTorch 2.6.0+cu124 |
| GPU | NVIDIA RTX 2050 (4.29 GB VRAM) |
| Mixed precision | `torch.amp.autocast` + `GradScaler` |
| Data loading | `torch.utils.data.DataLoader` (num_workers=0 for Windows) |
| Augmentation | `torchvision.transforms.functional` |
| Image I/O | Pillow 12.3.0 + OpenCV 5.0.0 |
| Application | Streamlit 1.66.0 |
| Testing | pytest 9.1.1 (32 tests) |
| Seed | 42 (random, numpy, torch, cuda) |

---

## 9. Experimental Setup

| Parameter | FCN | U-Net |
|-----------|-----|-------|
| Input size | 256 × 256 | 256 × 256 |
| Batch size | 8 | 8 |
| Epochs | 30 | 50 |
| Learning rate | 1e-3 | 1e-4 |
| LR patience | 5 | 5 |
| Early stop | 10 | 15 |

---

## 10. Results and Comparison

### 10.1 Training Curves
*(Refer to `outputs/eval_fcn/loss_curve.png` and `outputs/eval_unet/loss_curve.png`)*

### 10.2 Threshold Selection
Optimal binarization threshold was determined on the validation set:

| Model | Best threshold | Validation Dice |
|-------|----------------|-----------------|
| FCN Baseline | 0.30 | 0.7578 |
| U-Net | 0.35 | 0.7811 |

### 10.3 Test Set Metrics (111 held-out images)

| Metric | FCN Baseline | U-Net | Improvement |
|--------|-------------|-------|-------------|
| **Dice** | 0.7363 | **0.7680** | **+0.0317** (+3.17%) |
| **IoU** | 0.5827 | **0.6234** | **+0.0407** (+4.07%) |
| Precision | 0.7164 | **0.7636** | **+0.0472** (+4.72%) |
| Recall | 0.7574 | **0.7725** | **+0.0151** (+1.51%) |
| Pixel accuracy | 0.9893 | 0.9908 | +0.0015 |
| Test loss | 0.1665 | **0.1473** | -0.0192 |
| Inference (ms/image) | 0.54 ms | 2.13 ms | — |

### 10.4 Visual Comparison
*[Prediction grid — see `outputs/eval_*/predictions.png`]*

---

## 11. Error Analysis

**Observed failure modes:**

1. **Missed thin cracks (false negatives):** Cracks narrower than ~3 pixels are often below the model's effective resolution. These contribute to lower recall.

2. **Pavement texture false positives:** Joint lines, aggregate boundaries, and shadows can resemble crack patterns. FCN is more susceptible due to absence of fine-detail skip connections.

3. **Boundary imprecision:** Predicted crack regions are typically slightly wider or shorter than ground truth, because U-Net's decoder cannot perfectly recover subpixel-accurate boundaries from 256 × 256 resolution features.

4. **Edge-of-image artifacts:** Rotation augmentation introduces black padding at corners, which occasionally confuses the model near image borders.

---

## 12. Limitations

- **Hardware constraint:** 4.29 GB VRAM limits model depth and batch size. A pretrained ResNet-34 U-Net encoder would likely achieve higher Dice but exceeds available memory.
- **Dataset size:** ~3,700 training images is moderate. Performance on unseen crack types (alligator cracking, spalling) is unknown.
- **2D only:** The model processes individual frames with no temporal context.
- **Fixed threshold:** A per-image adaptive threshold would improve results for images with different illumination.
- **No post-processing:** Morphological thinning and connected-component filtering could improve boundary precision.
- **Experimental disclaimer:** Predictions have not been validated against professional engineering standards.

---

## 13. Conclusion

This project demonstrated pixel-level road crack segmentation using deep learning. A U-Net model with combined BCE+Dice loss was trained on ~3,700 annotated images and evaluated on a held-out test set. Key findings:

- U-Net's skip connections are critical for reconstructing thin crack structures that a plain FCN encoder-decoder cannot recover.
- Combined BCE + Dice loss is essential for handling ~98 % background class imbalance.
- Dice and IoU are the correct primary metrics; pixel accuracy is misleading and reported for completeness only.
- The pipeline is fully reproducible with fixed random seeds and saved checkpoints.

---

## 14. Future Improvements

1. Pretrained encoder (ResNet-34 or EfficientNet-B2) via transfer learning
2. Test-time augmentation (TTA) for more stable predictions
3. Larger, more diverse dataset including wet roads, night conditions, and varied crack types
4. Post-processing: morphological thinning, connected-components, length/width estimation
5. Real-time inference optimisation (model quantisation, TorchScript export)
6. Multi-class segmentation (longitudinal, transverse, alligator crack classification)

---

## 15. References

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. CVPR. DOI: 10.1109/CVPR.2015.7298965
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI. DOI: 10.1007/978-3-319-24574-4_28
3. Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). Road crack detection using deep convolutional neural network. ICIP. DOI: 10.1109/ICIP.2016.7533052
4. Zou, Q., et al. (2019). DeepCrack: Learning hierarchical convolutional features for crack detection. IEEE TIP, 28(3). DOI: 10.1109/TIP.2018.2878966
5. Yang, F., et al. (2019). Feature pyramid and hierarchical boosting network for pavement crack detection. IEEE TITS, 21(4). DOI: 10.1109/TITS.2019.2910595
6. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. 3DV. DOI: 10.1109/3DV.2016.79
7. He, K., et al. (2016). Deep residual learning for image recognition. CVPR. DOI: 10.1109/CVPR.2016.90
8. Lin, T.-Y., et al. (2017). Focal loss for dense object detection. ICCV. DOI: 10.1109/ICCV.2017.324
