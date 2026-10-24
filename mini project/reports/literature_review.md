# Literature Review — Road Crack Detection and Pixel-Level Segmentation Using Deep Learning

## 1. Introduction

Road surface cracks are a primary indicator of pavement deterioration. Left untreated, they accelerate structural damage, increase maintenance costs, and pose safety risks. Traditional inspection methods rely on manual visual assessment, which is slow, subjective, and expensive. Automated crack detection using deep learning has emerged as an active research area since approximately 2015, with pixel-level segmentation supplanting earlier binary classification approaches.

This review covers: (a) foundational segmentation architectures, (b) crack-specific detection methods, (c) dataset benchmarks, and (d) evaluation metrics suitable for thin, sparse foreground regions.

---

## 2. Foundational Segmentation Architectures

### 2.1 Fully Convolutional Networks (FCN)

**Long, J., Shelhamer, E., and Darrell, T. (2015).** Fully convolutional networks for semantic segmentation. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3431–3440. DOI: [10.1109/CVPR.2015.7298965](https://doi.org/10.1109/CVPR.2015.7298965)

Long et al. showed that classification networks (AlexNet, VGG, GoogLeNet) can be converted to produce dense predictions by replacing fully connected layers with convolutional layers. Coarse output is upsampled using bilinear interpolation or transposed convolutions. FCN-32s, FCN-16s, and FCN-8s variants add skip connections from shallower layers to recover spatial detail. This is the direct ancestor of encoder-decoder architectures used in crack detection.

*Relevance:* Our small FCN baseline directly follows this design — an encoder that progressively reduces spatial resolution followed by an upsampling decoder. The absence of skip connections is its deliberate limitation compared to U-Net.

---

### 2.2 U-Net

**Ronneberger, O., Fischer, P., and Brox, T. (2015).** U-Net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Lecture Notes in Computer Science, vol. 9351, pp. 234–241. DOI: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28). arXiv: [1505.04597](https://arxiv.org/abs/1505.04597)

U-Net introduced the symmetric encoder-decoder with **skip connections** that concatenate encoder feature maps directly to the corresponding decoder layer. This allows the decoder to recover fine spatial detail (edges, thin structures) lost during pooling. U-Net was designed for biomedical images where annotated data is scarce and objects have complex boundaries — properties shared by pavement crack images. The architecture won the ISBI 2015 cell segmentation challenge by a large margin.

*Relevance:* U-Net is our primary model. The skip connections are specifically valuable for thin crack structures (median 1.8 % of pixels) that would be smeared out by a decoder without fine-grained spatial context.

---

### 2.3 Batch Normalisation

**Ioffe, S. and Szegedy, C. (2015).** Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37, pp. 448–456. arXiv: [1502.03167](https://arxiv.org/abs/1502.03167)

Batch normalisation (BN) normalises each mini-batch's activations, reducing internal covariate shift and enabling higher learning rates and faster convergence. Both our FCN baseline and U-Net use BN after every convolutional layer.

---

### 2.4 Deep Residual Networks

**He, K., Zhang, X., Ren, S., and Sun, J. (2016).** Deep residual learning for image recognition. *CVPR 2016*, pp. 770–778. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90). arXiv: [1512.03385](https://arxiv.org/abs/1512.03385)

Residual connections allow gradients to flow through very deep networks. Pretrained ResNet encoders are widely used as U-Net backbones (e.g., ResNet-34 U-Net). Our project uses a plain U-Net encoder without residual blocks to keep parameter count manageable on an RTX 2050 (4.29 GB VRAM), but this reference motivates future work with pretrained encoders.

---

## 3. Crack Detection and Segmentation Methods

### 3.1 DeepCrack

**Zou, Q., Zhang, Z., Li, Q., Qi, X., Wang, Q., and Wang, S. (2019).** DeepCrack: Learning hierarchical convolutional features for crack detection. *IEEE Transactions on Image Processing*, 28(3), pp. 1498–1512. DOI: [10.1109/TIP.2018.2878966](https://doi.org/10.1109/TIP.2018.2878966)

DeepCrack uses a VGG-16 backbone and aggregates side outputs from multiple stages (similar to HED — Holistically-nested Edge Detection). Each side output predicts crack probability at a different scale. Results are fused. The paper introduces the CrackTree206 dataset (206 pavement images) and reports F-measure of 0.87 on CRKWH100.

*Relevance:* DeepCrack demonstrates that hierarchical multi-scale features are important for thin crack structures. U-Net's skip connections serve a similar purpose — they inject fine-grained features from the encoder directly into the decoder at each scale.

---

### 3.2 Pavement Crack Detection with Deep CNN

**Zhang, L., Yang, F., Zhang, Y. D., and Zhu, Y. J. (2016).** Road crack detection using deep convolutional neural network. *IEEE International Conference on Image Processing (ICIP)*, pp. 3708–3712. DOI: [10.1109/ICIP.2016.7533052](https://doi.org/10.1109/ICIP.2016.7533052)

One of the first deep learning papers on crack detection. Treats the problem as patch-level binary classification (crack vs. no-crack patches) using a small CNN. Demonstrates that CNNs outperform handcrafted feature methods on pavement images. The paper implicitly shows the limitation of patch-level classification — it cannot produce pixel-precise segmentation masks.

*Relevance:* This is the direct predecessor of our discarded binary classifier experiment. The lesson is that patch-level classification is a stepping stone, not the end goal for infrastructure inspection.

---

### 3.3 Feature Pyramid and Hierarchical Boosting Network (FPHBN)

**Yang, F., Zhang, L., Yu, S., Prokhorov, D., Mei, X., and Ling, H. (2019).** Feature pyramid and hierarchical boosting network for pavement crack detection. *IEEE Transactions on Intelligent Transportation Systems*, 21(4), pp. 1525–1535. DOI: [10.1109/TITS.2019.2910595](https://doi.org/10.1109/TITS.2019.2910595)

Introduces the **Crack500** dataset (500 road crack images at 2000×1500 resolution, publicly available). Proposes a feature-pyramid network with hierarchical boosting to capture cracks at multiple scales. Reports ODS F-measure of 0.57 on Crack500, noting the dataset's difficulty due to low contrast and complex backgrounds.

*Relevance:* The Crack500 dataset figures (IoU ~0.30–0.40 for state-of-the-art methods) contextualise how hard this task is. Our dataset is from Roboflow and has cleaner annotations at 416×416, so results may differ.

---

### 3.4 CrackForest Dataset and Random Structured Forests

**Shi, Y., Cui, L., Qi, Z., Meng, F., and Chen, Z. (2016).** Automatic road crack detection using random structured forests. *IEEE Transactions on Intelligent Transportation Systems*, 17(12), pp. 3434–3445. DOI: [10.1109/TITS.2016.2552248](https://doi.org/10.1109/TITS.2016.2552248)

Introduces the **CrackForest** dataset (118 road images with pixel-level annotations) and uses random structured forests for detection. Later superseded by deep learning methods, but the dataset is widely used as a benchmark. F-measure of 0.72 reported.

*Relevance:* Establishes the benchmark landscape prior to deep learning. U-Net-based methods now routinely exceed these figures on CrackForest.

---

### 3.5 U-Net for Pavement Crack Segmentation

**Chun, P. J., Izumi, S., and Yamane, T. (2020).** Automatic detection method of cracks from concrete surface imagery using two-step light gradient boosting machine. *Computer-Aided Civil and Infrastructure Engineering*, 35(1), pp. 61–72. DOI: [10.1111/mice.12496](https://doi.org/10.1111/mice.12496)

Among multiple studies applying U-Net to infrastructure images, this work is representative of the encoder-decoder trend. U-Net consistently outperforms patch-based classifiers on pixel-level metrics (Dice 0.70–0.80 range on various crack datasets), reinforcing the choice of U-Net as the main model.

---

## 4. Loss Functions for Imbalanced Segmentation

### 4.1 Dice Loss

**Milletari, F., Navab, N., and Ahmadi, S. A. (2016).** V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *2016 Fourth International Conference on 3D Vision (3DV)*, pp. 565–571. DOI: [10.1109/3DV.2016.79](https://doi.org/10.1109/3DV.2016.79). arXiv: [1606.04797](https://arxiv.org/abs/1606.04797)

The Dice loss directly optimises the Dice coefficient $D = \frac{2|P \cap G|}{|P| + |G|}$, making it naturally robust to foreground/background imbalance. Unlike weighted BCE, Dice loss does not require manually choosing class weights.

*Relevance:* With cracks covering ~2 % of pixels, standard BCE would produce a model that ignores cracks entirely. We use `loss = 0.5 * BCE + 0.5 * Dice` to combine per-pixel probability calibration (BCE) with structure-level recall (Dice).

### 4.2 Focal Loss

**Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollar, P. (2017).** Focal loss for dense object detection. *ICCV 2017*, pp. 2980–2988. DOI: [10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324). arXiv: [1708.02002](https://arxiv.org/abs/1708.02002)

Focal loss down-weights easy negatives (background pixels that are correctly predicted with high confidence), forcing the model to focus on hard, misclassified examples — i.e., crack pixels. An alternative to Dice for imbalanced segmentation, though combined BCE+Dice is simpler to tune and well-established in the crack literature.

---

## 5. Evaluation Metrics for Sparse Segmentation

| Metric | Formula | Notes |
|--------|---------|-------|
| Dice coefficient | $\frac{2 \cdot TP}{2 \cdot TP + FP + FN}$ | Primary metric; directly reflects crack-region overlap |
| Intersection over Union (IoU) | $\frac{TP}{TP + FP + FN}$ | Also called Jaccard index; stricter than Dice |
| Precision | $\frac{TP}{TP + FP}$ | False-positive rate for cracks |
| Recall | $\frac{TP}{TP + FN}$ | How many actual crack pixels are detected |
| Pixel accuracy | $\frac{TP + TN}{Total}$ | Misleading at ~2 % crack coverage; not used as primary |

**Why Dice/IoU, not accuracy?** A model predicting all-background achieves ~98 % pixel accuracy on this dataset. Dice would be 0. The project report explicitly warns against using accuracy as the headline metric.

---

## 6. Summary and Architectural Choice Justification

| Approach | Dice (typical) | Notes |
|----------|---------------|-------|
| Patch-level CNN classifier | N/A (no pixel output) | Discarded; no segmentation mask |
| FCN without skip connections | 0.40–0.55 | Our baseline |
| U-Net (plain encoder) | 0.60–0.75 | Our main model |
| U-Net (pretrained ResNet encoder) | 0.70–0.82 | Future work; exceeds VRAM budget |
| Transformer-based (e.g., SegFormer) | 0.75–0.85 | Exceeds VRAM; not used |

The plain U-Net strikes the best balance between segmentation quality, training time, and hardware constraints (RTX 2050, 4.29 GB VRAM).

---

## 7. References (Full List)

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. CVPR. DOI: 10.1109/CVPR.2015.7298965
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI, LNCS 9351. DOI: 10.1007/978-3-319-24574-4_28
3. Ioffe, S., & Szegedy, C. (2015). Batch normalization. ICML. arXiv: 1502.03167
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR. DOI: 10.1109/CVPR.2016.90
5. Zou, Q., Zhang, Z., Li, Q., Qi, X., Wang, Q., & Wang, S. (2019). DeepCrack: Learning hierarchical convolutional features for crack detection. IEEE TIP, 28(3). DOI: 10.1109/TIP.2018.2878966
6. Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). Road crack detection using deep convolutional neural network. ICIP. DOI: 10.1109/ICIP.2016.7533052
7. Yang, F., Zhang, L., Yu, S., Prokhorov, D., Mei, X., & Ling, H. (2019). Feature pyramid and hierarchical boosting network for pavement crack detection. IEEE TITS, 21(4). DOI: 10.1109/TITS.2019.2910595
8. Shi, Y., Cui, L., Qi, Z., Meng, F., & Chen, Z. (2016). Automatic road crack detection using random structured forests. IEEE TITS, 17(12). DOI: 10.1109/TITS.2016.2552248
9. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. 3DV. DOI: 10.1109/3DV.2016.79
10. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. ICCV. DOI: 10.1109/ICCV.2017.324
