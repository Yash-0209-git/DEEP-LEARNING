"""
config.py — Central configuration for Road Crack Detection project.

Import this module wherever you need paths or hyperparameters.
All paths are derived from the project root — no hardcoded usernames.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
# Two levels above src/config.py
ROOT = Path(__file__).resolve().parents[1]

# ── Data paths ─────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data" / "raw"
IMAGE_DIR  = DATA_DIR / "images"
MASK_DIR   = DATA_DIR / "masks"

# ── Output paths ───────────────────────────────────────────────────────────────
OUTPUT_DIR    = ROOT / "outputs"
MODEL_DIR     = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

for _d in (OUTPUT_DIR, MODEL_DIR, REPORTS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Image settings ─────────────────────────────────────────────────────────────
IMAGE_SIZE   = 256          # Both height and width after resizing
IMAGE_MEAN   = (0.485, 0.456, 0.406)   # ImageNet mean (R, G, B)
IMAGE_STD    = (0.229, 0.224, 0.225)   # ImageNet std  (R, G, B)

# ── Training — shared ──────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 8
NUM_WORKERS  = 0            # Keep 0 for Windows (avoids multiprocessing spawn issues)

# ── Baseline FCN hyperparameters ───────────────────────────────────────────────
FCN_EPOCHS        = 30
FCN_LR            = 1e-3
FCN_LR_PATIENCE   = 5       # ReduceLROnPlateau patience
FCN_LR_FACTOR     = 0.5
FCN_EARLY_STOP    = 10      # Stop if val loss does not improve for this many epochs
FCN_BASE_CHANNELS = 32      # Width multiplier — first encoder block output channels

# ── U-Net hyperparameters ──────────────────────────────────────────────────────
UNET_EPOCHS        = 50
UNET_LR            = 1e-4
UNET_LR_PATIENCE   = 5
UNET_LR_FACTOR     = 0.5
UNET_EARLY_STOP    = 15
UNET_BASE_CHANNELS = 32     # [32, 64, 128, 256] encoder widths; bottleneck = 512

# ── Loss function ──────────────────────────────────────────────────────────────
# Combined loss = LOSS_BCE_WEIGHT * BCE + LOSS_DICE_WEIGHT * Dice
LOSS_BCE_WEIGHT  = 0.5
LOSS_DICE_WEIGHT = 0.5

# ── Evaluation ─────────────────────────────────────────────────────────────────
# Threshold sweep on validation set to choose the best operating threshold.
THRESHOLD_MIN    = 0.10
THRESHOLD_MAX    = 0.90
THRESHOLD_STEP   = 0.05
DEFAULT_THRESHOLD = 0.50    # Used before threshold tuning is complete

# ── Mixed precision ────────────────────────────────────────────────────────────
USE_AMP = True              # torch.amp — RTX 2050 supports fp16; saves VRAM
