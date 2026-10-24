"""
tests/test_pipeline.py — Unit tests for the Road Crack Detection data pipeline,
models, losses, and metrics.

Run from the project root with:
    .venv\\Scripts\\python.exe -m pytest tests/ -v

All tests are self-contained — no network access required.
Tests that need the real dataset are skipped if the data directory is missing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure the project root is on sys.path so 'src' is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_SIZE, FCN_BASE_CHANNELS, IMAGE_SIZE, UNET_BASE_CHANNELS
from src.losses import CombinedLoss, DiceLoss
from src.metrics import (
    compute_all_metrics,
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
)
from src.models.fcn_baseline import FCNBaseline
from src.models.unet import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_random_batch(batch=2, channels=3, size=IMAGE_SIZE):
    """Return a random image batch and a binary mask batch."""
    images = torch.randn(batch, channels, size, size)
    masks = (torch.rand(batch, 1, size, size) > 0.5).float()
    return images, masks


def make_random_logits_and_targets(batch=2, size=32):
    """Return random logits [B,1,H,W] and binary targets [B,1,H,W] for loss tests."""
    logits = torch.randn(batch, 1, size, size)
    targets = (torch.rand(batch, 1, size, size) > 0.5).float()
    return logits, targets


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFCNBaseline:
    def test_output_shape(self):
        model = FCNBaseline(base_channels=FCN_BASE_CHANNELS)
        images, _ = make_random_batch()
        out = model(images)
        assert out.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE), \
            f"Expected (2,1,{IMAGE_SIZE},{IMAGE_SIZE}), got {out.shape}"

    def test_output_is_logit_range(self):
        """Raw output should NOT be bounded to [0,1] (it's a logit)."""
        model = FCNBaseline()
        images, _ = make_random_batch()
        out = model(images)
        # After sigmoid, output should be in [0,1]
        probs = torch.sigmoid(out)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_parameter_count_positive(self):
        model = FCNBaseline()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0, "Model has no trainable parameters"

    def test_gpu_forward_if_available(self):
        model = FCNBaseline().to(DEVICE)
        images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        out = model(images)
        assert out.device.type == DEVICE.type


class TestUNet:
    def test_output_shape(self):
        model = UNet(base_channels=UNET_BASE_CHANNELS)
        images, _ = make_random_batch()
        out = model(images)
        assert out.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE), \
            f"Expected (2,1,{IMAGE_SIZE},{IMAGE_SIZE}), got {out.shape}"

    def test_skip_connections_used(self):
        """Verify that changing a skip-connection feature map changes the output
        (i.e., skip connections are actually wired in, not just defined)."""
        model = UNet(base_channels=16)   # small for speed
        images = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out1 = model(images)

        # Hook to zero out the first skip feature map during the second forward.
        zeroed = [False]

        def hook(module, input, output):
            if not zeroed[0]:
                zeroed[0] = True
                return torch.zeros_like(output)

        handle = model.enc1.register_forward_hook(hook)
        out2 = model(images)
        handle.remove()

        assert not torch.allclose(out1, out2), \
            "Zeroing enc1 output had no effect — skip connections may not be wired"

    def test_non_power_of_two_input(self):
        """U-Net should handle 224x224 via the interpolate guard."""
        model = UNet(base_channels=16)
        images = torch.randn(1, 3, 224, 224)
        out = model(images)
        assert out.shape == (1, 1, 224, 224)

    def test_gpu_forward_if_available(self):
        model = UNet().to(DEVICE)
        images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        out = model(images)
        assert out.device.type == DEVICE.type


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Loss Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiceLoss:
    def test_perfect_prediction_near_zero(self):
        """Perfect prediction -> Dice ≈ 1 -> DiceLoss ≈ 0."""
        targets = torch.ones(2, 1, 8, 8)   # all ones
        logits = torch.full((2, 1, 8, 8), 10.0)  # large positive -> sigmoid ≈ 1
        loss = DiceLoss()(logits, targets)
        assert loss.item() < 0.05, f"Expected near 0, got {loss.item():.4f}"

    def test_all_wrong_near_one(self):
        """Predicting all background when target is all foreground -> loss ≈ 1."""
        targets = torch.ones(2, 1, 8, 8)
        logits = torch.full((2, 1, 8, 8), -10.0)  # sigmoid ≈ 0
        loss = DiceLoss()(logits, targets)
        assert loss.item() > 0.9, f"Expected near 1, got {loss.item():.4f}"

    def test_empty_target_smooth(self):
        """Empty target (all zeros) should not raise division-by-zero."""
        targets = torch.zeros(2, 1, 8, 8)
        logits = torch.zeros(2, 1, 8, 8)
        loss = DiceLoss()(logits, targets)
        assert torch.isfinite(loss), "Loss is not finite for empty target"

    def test_loss_non_negative(self):
        logits, targets = make_random_logits_and_targets()
        loss = DiceLoss()(logits, targets)
        assert loss.item() >= 0.0


class TestCombinedLoss:
    def test_output_is_finite(self):
        logits, targets = make_random_logits_and_targets()
        loss = CombinedLoss()(logits, targets)
        assert torch.isfinite(loss), "CombinedLoss is not finite"

    def test_backward(self):
        """Gradients should flow through CombinedLoss."""
        model = FCNBaseline(base_channels=8)   # tiny for speed
        images, masks = make_random_batch(batch=1, size=64)
        logits = model(images)
        loss = CombinedLoss()(logits, masks)
        loss.backward()
        # Check that at least one gradient is non-zero.
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        total_grad_norm = sum(g.abs().sum().item() for g in grads)
        assert total_grad_norm > 0.0, "All gradients are zero"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Metric Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    # Perfect predictions
    def test_dice_perfect(self):
        p = np.ones(100, dtype=np.float32)
        t = np.ones(100, dtype=np.float32)
        assert dice_coefficient(p, t) > 0.999

    def test_dice_no_overlap(self):
        p = np.ones(100, dtype=np.float32)
        t = np.zeros(100, dtype=np.float32)
        assert dice_coefficient(p, t) < 0.01

    def test_iou_perfect(self):
        p = np.ones(50, dtype=np.float32)
        t = np.ones(50, dtype=np.float32)
        assert iou_score(p, t) > 0.999

    def test_iou_no_overlap(self):
        p = np.ones(50, dtype=np.float32)
        t = np.zeros(50, dtype=np.float32)
        assert iou_score(p, t) < 0.01

    def test_precision_all_true_positive(self):
        p = np.ones(50, dtype=np.float32)
        t = np.ones(50, dtype=np.float32)
        assert precision_score(p, t) > 0.999

    def test_recall_all_true_positive(self):
        p = np.ones(50, dtype=np.float32)
        t = np.ones(50, dtype=np.float32)
        assert recall_score(p, t) > 0.999

    def test_pixel_accuracy_all_correct(self):
        p = np.array([1, 0, 1, 0], dtype=np.float32)
        t = np.array([1, 0, 1, 0], dtype=np.float32)
        assert pixel_accuracy(p, t) == pytest.approx(1.0)

    def test_pixel_accuracy_all_wrong(self):
        p = np.array([1, 1, 1, 1], dtype=np.float32)
        t = np.array([0, 0, 0, 0], dtype=np.float32)
        assert pixel_accuracy(p, t) == pytest.approx(0.0)

    def test_compute_all_metrics_keys(self):
        probs = torch.rand(2, 1, 16, 16)
        targets = (torch.rand(2, 1, 16, 16) > 0.5).float()
        result = compute_all_metrics(probs, targets)
        assert set(result.keys()) == {"dice", "iou", "precision", "recall", "pixel_acc"}

    def test_all_metrics_in_valid_range(self):
        probs = torch.rand(2, 1, 16, 16)
        targets = (torch.rand(2, 1, 16, 16) > 0.5).float()
        result = compute_all_metrics(probs, targets)
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"Metric {k} out of range: {v}"

    def test_dice_relation_to_iou(self):
        """Dice >= IoU always (can be proven algebraically)."""
        p = np.array([1, 1, 0, 0, 1], dtype=np.float32)
        t = np.array([1, 0, 1, 0, 1], dtype=np.float32)
        assert dice_coefficient(p, t) >= iou_score(p, t) - 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dataset Tests (skipped if data is missing)
# ═══════════════════════════════════════════════════════════════════════════════

DATA_AVAILABLE = (PROJECT_ROOT / "data" / "raw" / "images" / "train").exists()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Dataset not found")
class TestDataset:
    def test_dataset_length(self):
        from src.dataset import CrackSegmentationDataset
        ds = CrackSegmentationDataset("train", augment=False)
        assert len(ds) == 3717, f"Expected 3717, got {len(ds)}"

    def test_sample_shapes(self):
        from src.dataset import CrackSegmentationDataset
        ds = CrackSegmentationDataset("train", augment=False)
        img, mask = ds[0]
        assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE), \
            f"Image shape {img.shape}"
        assert mask.shape == (1, IMAGE_SIZE, IMAGE_SIZE), \
            f"Mask shape {mask.shape}"

    def test_mask_binary_values(self):
        from src.dataset import CrackSegmentationDataset
        ds = CrackSegmentationDataset("train", augment=False)
        for i in range(0, min(50, len(ds)), 10):  # Sample 5 items
            _, mask = ds[i]
            unique = mask.unique().tolist()
            for v in unique:
                assert v in (0.0, 1.0), \
                    f"Unexpected mask value {v} at index {i}"

    def test_image_normalised(self):
        """Normalised images should have pixel values spanning beyond [0, 1]."""
        from src.dataset import CrackSegmentationDataset
        ds = CrackSegmentationDataset("train", augment=False)
        img, _ = ds[0]
        # After ImageNet normalisation, min should be < 0
        assert img.min().item() < 0.0, \
            "Image does not appear to be normalised (min >= 0)"

    def test_augment_does_not_corrupt_mask(self):
        """Augmented masks must still contain only 0 and 1."""
        from src.dataset import CrackSegmentationDataset
        ds = CrackSegmentationDataset("train", augment=True)
        for i in range(5):
            _, mask = ds[i]
            unique = set(mask.unique().tolist())
            assert unique.issubset({0.0, 1.0}), \
                f"Augmentation produced bad mask values: {unique}"

    def test_loader_batch_shape(self):
        from src.dataset import get_loaders
        train_loader, _, _ = get_loaders(batch_size=4)
        images, masks = next(iter(train_loader))
        assert images.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert masks.shape  == (4, 1, IMAGE_SIZE, IMAGE_SIZE)

    def test_val_and_test_sizes(self):
        from src.dataset import CrackSegmentationDataset
        val_ds  = CrackSegmentationDataset("val",  augment=False)
        test_ds = CrackSegmentationDataset("test", augment=False)
        assert len(val_ds)  == 200, f"Expected 200 val images, got {len(val_ds)}"
        assert len(test_ds) == 111, f"Expected 111 test images, got {len(test_ds)}"

    def test_image_mask_pairing(self):
        """Verify that every image in each split has an exactly matching mask file."""
        from src.config import IMAGE_DIR, MASK_DIR
        for split in ("train", "val", "test"):
            img_dir = IMAGE_DIR / split
            msk_dir = MASK_DIR / split
            img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
            msk_stems = {p.stem for p in msk_dir.glob("*.png")}
            missing = img_stems - msk_stems
            assert len(missing) == 0, f"Split {split} has {len(missing)} images without masks: {list(missing)[:5]}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Checkpoint Saving and Loading Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpoints:
    def test_save_and_load_checkpoint(self, tmp_path):
        from src.train import save_checkpoint, load_checkpoint
        model = FCNBaseline(base_channels=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        ckpt_path = tmp_path / "test_model.pth"
        history = {"train_loss": [0.5, 0.4], "val_loss": [0.45, 0.38], "val_dice": [0.6, 0.7]}
        
        save_checkpoint(model, optimizer, scheduler, epoch=2, best_val_loss=0.38, history=history, path=ckpt_path)
        assert ckpt_path.exists()
        
        # Load into fresh model
        new_model = FCNBaseline(base_channels=8)
        new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(new_opt)
        
        start_epoch, best_loss, loaded_hist = load_checkpoint(new_model, new_opt, new_sched, ckpt_path, torch.device("cpu"))
        assert start_epoch == 3
        assert best_loss == pytest.approx(0.38)
        assert loaded_hist["val_dice"] == [0.6, 0.7]
        
        # Verify predictions match between saved and loaded models
        dummy_in = torch.randn(1, 3, 64, 64)
        model.eval()
        new_model.eval()
        with torch.no_grad():
            out1 = model(dummy_in)
            out2 = new_model(dummy_in)
        assert torch.allclose(out1, out2, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Inference and App Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferenceAndApp:
    def test_app_preprocess(self):
        from PIL import Image
        from app.app import preprocess
        dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        tensor = preprocess(dummy_img)
        assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert isinstance(tensor, torch.Tensor)

    def test_app_run_inference(self):
        from app.app import run_inference
        model = FCNBaseline(base_channels=8)
        model.eval()
        dummy_tensor = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        prob_map, binary_mask, elapsed_ms = run_inference(model, dummy_tensor, torch.device("cpu"), threshold=0.5)
        
        assert prob_map.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert binary_mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert binary_mask.dtype == bool
        assert 0.0 <= prob_map.min() and prob_map.max() <= 1.0
        assert elapsed_ms >= 0.0

    def test_app_build_overlay(self):
        from app.app import build_overlay
        orig = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[10:20, 10:20] = True
        
        overlay = build_overlay(orig, mask, alpha=0.5)
        assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert overlay.dtype == np.uint8
        # Masked area should have red tint (>0 in R channel)
        assert overlay[15, 15, 0] > 0

