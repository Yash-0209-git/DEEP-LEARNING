"""
dataset.py — PyTorch Dataset and DataLoader factory for Road Crack Detection.

Loads RGB images and binary segmentation masks. Applies:
  - Resize to IMAGE_SIZE x IMAGE_SIZE.
  - ImageNet normalisation for images.
  - Nearest-neighbour resize for masks (no interpolation artefacts).
  - Training-only spatial augmentations applied identically to image and mask.
  - Colour jitter on image only (not mask).

Usage:
    from src.dataset import get_loaders
    train_loader, val_loader, test_loader = get_loaders()
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.config import (
    BATCH_SIZE,
    IMAGE_DIR,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    MASK_DIR,
    NUM_WORKERS,
    SEED,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class CrackSegmentationDataset(Dataset):
    """
    Returns (image_tensor, mask_tensor) pairs.

    image_tensor : FloatTensor [3, H, W] — normalised RGB.
    mask_tensor  : FloatTensor [1, H, W] — binary, values {0.0, 1.0}.
    """

    def __init__(self, split: str, augment: bool = False):
        """
        Args:
            split:   One of 'train', 'val', 'test'.
            augment: If True, apply random spatial + colour augmentations.
                     Should only be True for the training split.
        """
        self.split = split
        self.augment = augment
        self.image_dir = IMAGE_DIR / split
        self.mask_dir = MASK_DIR / split

        # Collect all image paths that have a corresponding mask.
        self.image_paths: list[Path] = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
            and (self.mask_dir / f"{p.stem}.png").exists()
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No image-mask pairs found for split '{split}'. "
                f"Checked: {self.image_dir}"
            )

        # Normalisation transform (image only — applied after tensor conversion).
        self.normalise = transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        # Load image as RGB PIL Image.
        image = Image.open(img_path).convert("RGB")
        # Load mask as grayscale PIL Image (values 0 or 255).
        mask = Image.open(mask_path).convert("L")

        # ── Resize ────────────────────────────────────────────────────────────
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)

        # ── Training augmentations ────────────────────────────────────────────
        if self.augment:
            image, mask = self._augment(image, mask)

        # ── Convert to tensors ────────────────────────────────────────────────
        # Image: PIL -> [0,1] float tensor [3, H, W]
        image_tensor = TF.to_tensor(image)

        # Mask: PIL grayscale -> numpy -> binary {0,1} float tensor [1, H, W]
        mask_array = np.array(mask, dtype=np.float32)
        mask_binary = (mask_array > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]

        # ── Normalise image ───────────────────────────────────────────────────
        image_tensor = self.normalise(image_tensor)

        return image_tensor, mask_tensor

    def _augment(self, image: Image.Image, mask: Image.Image):
        """
        Apply identical spatial transforms to both image and mask.
        Colour jitter is applied to the image only.
        """
        # Horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Vertical flip
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation ±15 degrees (nearest for mask to avoid blending values)
        if random.random() < 0.5:
            angle = random.uniform(-15.0, 15.0)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Colour jitter — image only, never mask
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)

        return image, mask


def get_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader).

    Augmentation is applied only to the training split.
    A fixed generator seed is set for reproducibility.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_ds = CrackSegmentationDataset("train", augment=True)
    val_ds = CrackSegmentationDataset("val", augment=False)
    test_ds = CrackSegmentationDataset("test", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check — run with: python -m src.dataset
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Train batches : {len(train_loader)}  ({len(train_loader.dataset)} images)")
    print(f"Val   batches : {len(val_loader)}  ({len(val_loader.dataset)} images)")
    print(f"Test  batches : {len(test_loader)}  ({len(test_loader.dataset)} images)")

    images, masks = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  images shape : {images.shape}  dtype={images.dtype}")
    print(f"  masks  shape : {masks.shape}   dtype={masks.dtype}")
    print(f"  image  min/max: {images.min():.3f} / {images.max():.3f}")
    print(f"  mask   unique values: {masks.unique().tolist()}")
