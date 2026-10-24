"""
metrics.py — Segmentation evaluation metrics for Road Crack Detection.

All functions operate on CPU tensors or numpy arrays.
They are designed to be called per-batch and aggregated, OR on full-dataset
accumulated arrays.

Primary metrics:
    dice_coefficient : overlap measure; handles imbalance; primary ranking metric.
    iou_score        : Jaccard index; stricter than Dice; secondary ranking.
    precision        : TP / (TP + FP) — false-positive rate.
    recall           : TP / (TP + FN) — false-negative rate; crack detection sensitivity.

All functions accept:
    preds   : BoolTensor or np.ndarray — thresholded binary predictions {0, 1}.
    targets : BoolTensor or np.ndarray — ground-truth binary masks {0, 1}.

and return a float scalar.
"""

import numpy as np
import torch

SMOOTH = 1e-6   # Prevents zero-division on all-background images.


def _to_numpy(arr) -> np.ndarray:
    """Convert tensor or array to a flattened float32 numpy array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float32).ravel()


def dice_coefficient(preds, targets) -> float:
    """
    Dice coefficient = 2*TP / (2*TP + FP + FN).

    Returns a value in [0, 1].  1 = perfect overlap.
    """
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    intersection = (p * t).sum()
    return float((2.0 * intersection + SMOOTH) / (p.sum() + t.sum() + SMOOTH))


def iou_score(preds, targets) -> float:
    """
    Intersection over Union = TP / (TP + FP + FN).

    Returns a value in [0, 1].  1 = perfect overlap.
    """
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    intersection = (p * t).sum()
    union = p.sum() + t.sum() - intersection
    return float((intersection + SMOOTH) / (union + SMOOTH))


def precision_score(preds, targets) -> float:
    """
    Precision = TP / (TP + FP).

    High precision means few background pixels are incorrectly labelled as crack.
    """
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    return float((tp + SMOOTH) / (tp + fp + SMOOTH))


def recall_score(preds, targets) -> float:
    """
    Recall = TP / (TP + FN).

    High recall means most actual crack pixels are detected.
    For safety-critical infrastructure, recall is prioritised over precision.
    """
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    tp = (p * t).sum()
    fn = ((1.0 - p) * t).sum()
    return float((tp + SMOOTH) / (tp + fn + SMOOTH))


def pixel_accuracy(preds, targets) -> float:
    """
    Pixel accuracy = (TP + TN) / Total.

    WARNING: Misleading on imbalanced datasets (~98 % accuracy by predicting
    all-background on this crack dataset).  Reported for completeness only.
    """
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    correct = (p == t).sum()
    return float(correct / len(t))


def compute_all_metrics(preds, targets, threshold: float = 0.5) -> dict:
    """
    Convenience wrapper.

    Args:
        preds     : FloatTensor of probabilities [B, 1, H, W], or binary array.
        targets   : Binary ground truth  [B, 1, H, W].
        threshold : Applied to preds if they are probabilities.

    Returns:
        dict with keys: dice, iou, precision, recall, pixel_acc.
    """
    if isinstance(preds, torch.Tensor):
        preds = (preds >= threshold).float()

    return {
        "dice":      dice_coefficient(preds, targets),
        "iou":       iou_score(preds, targets),
        "precision": precision_score(preds, targets),
        "recall":    recall_score(preds, targets),
        "pixel_acc": pixel_accuracy(preds, targets),
    }
