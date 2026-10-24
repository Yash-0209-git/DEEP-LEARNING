"""
losses.py — Loss functions for binary crack segmentation.

Combined loss = BCE_WEIGHT * BCEWithLogitsLoss + DICE_WEIGHT * DiceLoss

BCEWithLogitsLoss: per-pixel probability calibration — numerically stable.
DiceLoss:          structure-level overlap — naturally handles class imbalance.

Both expect:
    logits  : FloatTensor [B, 1, H, W] — raw model output (before sigmoid).
    targets : FloatTensor [B, 1, H, W] — binary ground truth, values {0, 1}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LOSS_BCE_WEIGHT, LOSS_DICE_WEIGHT


class DiceLoss(nn.Module):
    """
    Soft Dice loss over a batch.

    dice = (2 * sum(p * t) + smooth) / (sum(p) + sum(t) + smooth)
    loss = 1 - dice

    Smooth (epsilon) prevents division by zero on empty masks.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions for the dot-product.
        probs_flat = probs.view(probs.size(0), -1)      # [B, H*W]
        targets_flat = targets.view(targets.size(0), -1) # [B, H*W]

        intersection = (probs_flat * targets_flat).sum(dim=1)          # [B]
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)        # [B]

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of BCEWithLogitsLoss and DiceLoss.

    loss = bce_weight * BCE(logits, targets) + dice_weight * Dice(logits, targets)
    """

    def __init__(
        self,
        bce_weight: float = LOSS_BCE_WEIGHT,
        dice_weight: float = LOSS_DICE_WEIGHT,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
