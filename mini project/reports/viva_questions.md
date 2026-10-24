# Viva Questions and Answers — Road Crack Detection

*Prepare these questions thoroughly. Answers are written for a college-level oral defence.*

---

## 1. General / Motivation

**Q: Why is road crack detection important?**  
A: Road cracks indicate pavement deterioration. If left untreated, small surface cracks admit water, accelerate freeze-thaw cycles, and lead to potholes or structural failure. Automated detection enables faster, cheaper, and more consistent inspection than manual methods, which rely on trained engineers walking roads — a slow, expensive, and subjective process.

**Q: Why use deep learning instead of traditional image processing?**  
A: Traditional methods (thresholding, edge detection, morphological operations) rely on hand-crafted rules that fail under variable lighting, surface texture, and crack shapes. Deep learning models learn features directly from labelled images and generalise across these variations. CNN-based methods consistently outperform handcrafted approaches on standard crack benchmarks since 2016.

**Q: Why is this a segmentation problem and not a classification problem?**  
A: Classification produces a single label per image ("crack present / absent"). Segmentation produces a label for every pixel, telling us exactly *where* cracks are. For infrastructure inspection, location and extent are as important as presence. Our earlier binary classifier experiment confirmed this: even with 100 % apparent accuracy, it produced no actionable spatial information.

---

## 2. Dataset

**Q: Where does the dataset come from?**  
A: The dataset was sourced from Roboflow via an NDJSON manifest. It contains 4,028 road/pavement images (416 × 416 px) split into 3,717 train, 200 validation, and 111 test images. Annotations are polygon segments that were rasterised into binary masks using `cv2.fillPoly`.

**Q: What was the class imbalance in your dataset?**  
A: Crack pixels make up roughly 2.2 % of each training image on average. The remaining 97.8 % are background. This is severe imbalance — a naive model predicting all-background would achieve ~98 % pixel accuracy while being completely useless.

**Q: How did you handle the class imbalance?**  
A: Two ways. First, we use a combined loss function (0.5 × BCE + 0.5 × Dice) instead of plain cross-entropy. Dice loss directly optimises pixel-overlap between prediction and ground truth, making it naturally robust to imbalance. Second, we use Dice and IoU as our primary evaluation metrics rather than pixel accuracy.

**Q: Did you check for data leakage between train/val/test?**  
A: Yes. We checked all 4,028 image file stems for overlap between splits. Zero duplicates were found. The splits are cleanly separated.

---

## 3. Preprocessing and Augmentation

**Q: Why did you resize images to 256 × 256 instead of keeping 416 × 416?**  
A: Memory constraints. The RTX 2050 has 4.29 GB of VRAM. At 416 × 416 with batch size 8, the U-Net's intermediate feature maps would exceed available VRAM. At 256 × 256, training runs comfortably with AMP enabled.

**Q: What augmentations did you apply and why?**  
A: Horizontal flip, vertical flip, and random rotation (±15°) — all applied identically to the image and its mask to maintain alignment. Colour jitter (brightness/contrast) is applied to the image only. We excluded colour inversion (destroys crack texture distinguishability) and elastic deformation (could artificially merge thin cracks).

**Q: Why use nearest-neighbour interpolation for masks during resize?**  
A: Bilinear interpolation would blend the binary 0/255 values, creating intermediate grey pixels. Since masks must be strictly binary (background or crack), nearest-neighbour is the only correct choice.

**Q: What normalisation did you apply?**  
A: We subtracted the ImageNet mean [0.485, 0.456, 0.406] and divided by the ImageNet standard deviation [0.229, 0.224, 0.225] channel-wise. This normalisation is standard practice when using ImageNet-pretrained statistics, and it matches the expected input distribution for most PyTorch models.

---

## 4. Model Architecture

**Q: Explain U-Net.**  
A: U-Net has a symmetric encoder-decoder structure. The encoder (left side) applies successive convolution + pooling blocks to progressively reduce spatial resolution while increasing channel depth — learning "what" features are present. The decoder (right side) upsamples back to the original resolution. The key innovation is **skip connections**: at each level, the encoder's feature map is concatenated to the corresponding decoder layer. This gives the decoder both high-level semantic context (from the bottleneck) and fine spatial detail (from the encoder), enabling it to reconstruct thin crack boundaries that would otherwise be lost during pooling.

**Q: What is the bottleneck in U-Net?**  
A: The deepest layer between encoder and decoder. In our implementation, it takes 256-channel features at 1/16 spatial resolution and produces 512 channels. It represents the most compressed, highest-level representation of the image — where global context is captured.

**Q: Why does U-Net outperform the FCN baseline?**  
A: The FCN baseline has no skip connections. Information from shallow layers (fine crack edges) is discarded during downsampling and cannot be recovered during upsampling. U-Net's skip connections preserve this spatial information, allowing the decoder to make sharper, more accurate predictions for thin crack structures.

**Q: What is the output of your model?**  
A: A single-channel logit map [B, 1, H, W]. After sigmoid, each pixel value is a probability in [0, 1] representing the likelihood of it being a crack pixel. We threshold this map at the best validation threshold to produce a binary mask.

---

## 5. Loss Function

**Q: Why not use standard cross-entropy loss?**  
A: Standard cross-entropy assigns equal weight to background and crack pixels. With 98 % background, the loss is dominated by correctly classified background pixels. The model learns to predict all-background to minimise loss, achieving high accuracy but zero Dice. We need a loss that specifically penalises missing crack pixels.

**Q: Explain Dice loss.**  
A: Dice loss is `1 - Dice_coefficient`, where `Dice = 2*TP / (2*TP + FP + FN)`. It directly measures overlap between prediction and target. Because it normalises by the foreground region size rather than total pixels, it gives equal importance to each crack regardless of what fraction of the image it covers. Mathematically: $L_{Dice} = 1 - \frac{2 \sum p_i t_i + \epsilon}{\sum p_i + \sum t_i + \epsilon}$ where $\epsilon$ is a smoothing constant.

**Q: Why combine BCE and Dice?**  
A: BCE provides per-pixel probability calibration — it ensures the model outputs meaningful confidence values. Dice ensures the model learns to overlap the entire crack region, not just nearby pixels. Together they balance pixel-level accuracy and structure-level recall. The 50/50 weighting is a common and empirically validated choice in the medical/crack segmentation literature.

---

## 6. Training

**Q: What is mixed-precision training (AMP)?**  
A: Automatic Mixed Precision uses 16-bit floats (fp16) for forward/backward computation and 32-bit floats for weight updates. This halves VRAM usage for activations, allowing larger batch sizes or models. PyTorch's `GradScaler` prevents fp16 underflow by scaling loss values. RTX 2050 supports Tensor Core fp16 acceleration.

**Q: How did you prevent overfitting?**  
A: Three mechanisms: (1) data augmentation increases effective training set size; (2) early stopping halts training if validation loss does not improve for 15 epochs; (3) we save the best checkpoint (by validation loss) rather than the last, ensuring we select the model at peak generalisation.

**Q: What is ReduceLROnPlateau?**  
A: A learning rate scheduler that reduces the learning rate by a factor (0.5) when the monitored metric (validation loss) has not improved for a set number of epochs (patience=5). This allows the model to make fine-grained updates when approaching a minimum, improving final convergence.

---

## 7. Evaluation

**Q: Why is pixel accuracy a misleading metric here?**  
A: Because of class imbalance. A model predicting all-background achieves ~98 % pixel accuracy on our dataset (98 % of pixels are background). This metric cannot distinguish a completely broken model from a well-trained one. Dice and IoU measure overlap of the foreground region only, so they correctly score the all-background predictor as 0.

**Q: Explain the Dice coefficient.**  
A: `Dice = 2*TP / (2*TP + FP + FN)`. It measures the overlap between predicted and actual crack regions, normalised to [0, 1]. A score of 1 means perfect overlap; 0 means no overlap. It is twice the intersection divided by the sum of both sets' sizes.

**Q: What is IoU (Intersection over Union)?**  
A: Also called the Jaccard index: `IoU = TP / (TP + FP + FN)`. It is always ≤ Dice. IoU is stricter — it penalises false positives and false negatives more than Dice does. Both are used; Dice is the primary ranking metric because it is more sensitive at low foreground coverage.

**Q: How did you select the classification threshold?**  
A: We swept thresholds from 0.10 to 0.90 in 0.05 steps on the **validation set** and selected the threshold that maximised Dice. This threshold is then fixed for all test-set evaluation. The test set is never used for threshold tuning — doing so would inflate reported results.

---

## 8. Application

**Q: How does the Streamlit app work?**  
A: The app loads the best checkpoint once at startup (cached across sessions). The user uploads a road image. The image is resized to 256 × 256, normalised, and passed through the model. The sigmoid output is thresholded to produce a binary mask. The app displays the original image, predicted mask, and a red overlay on top of the original. Inference time and crack coverage percentage are also shown.

**Q: Is the app suitable for real infrastructure inspection?**  
A: No. It is a research prototype trained on ~3,700 images. It has not been validated against professional inspection standards or on diverse real-world conditions. The app includes an explicit disclaimer to this effect.

---

## 9. Results and Analysis

**Q: What results did you expect and why?**  
A: We expected FCN Dice in the 0.40–0.55 range and U-Net Dice in the 0.60–0.75 range, consistent with literature reports on crack datasets of similar size and difficulty. U-Net should outperform FCN because skip connections allow it to recover fine crack detail.

**Q: If your model performs poorly (Dice < 0.5), what could be the reason?**  
A: Possible causes: (1) insufficient training epochs; (2) learning rate too high, causing unstable training; (3) the threshold is poorly chosen; (4) the model is overfitting to training data; (5) cracks in the test set differ significantly from training distribution. We would investigate by plotting loss curves, visualising predictions, and checking per-image metrics for outliers.

**Q: What are the main types of errors you observed?**  
A: (1) False negatives — thin, low-contrast cracks are missed, especially at boundaries. (2) False positives — pavement texture or shadows misclassified as cracks. (3) Boundary imprecision — predicted crack regions are slightly wider or shorter than ground truth.

---

## 10. Future Work

**Q: How would you improve this project?**  
A: (1) Use a ResNet-34 or EfficientNet pretrained encoder for U-Net — pretrained encoders significantly improve small-dataset performance. (2) Test-time augmentation (TTA) — average predictions over flipped/rotated versions for more stable outputs. (3) Larger dataset with diverse crack types (wet, dry, alligator, transverse). (4) Post-processing: connected-components analysis and morphological thinning to sharpen crack boundaries. (5) Multi-scale inference to handle cracks at different image resolutions.
