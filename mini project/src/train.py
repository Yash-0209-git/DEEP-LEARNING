"""
train.py — Training loop for Road Crack Detection segmentation models.

Usage (from project root, with .venv activated):

    Train FCN baseline:
        python -m src.train --model fcn

    Train U-Net:
        python -m src.train --model unet

    Resume from checkpoint:
        python -m src.train --model unet --resume

Options:
    --model   : 'fcn' or 'unet'  (required)
    --resume  : resume from the latest checkpoint for this model
    --epochs  : override number of epochs
    --lr      : override learning rate
    --batch   : override batch size
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    FCN_BASE_CHANNELS,
    FCN_EARLY_STOP,
    FCN_EPOCHS,
    FCN_LR,
    FCN_LR_FACTOR,
    FCN_LR_PATIENCE,
    OUTPUT_DIR,
    SEED,
    UNET_BASE_CHANNELS,
    UNET_EARLY_STOP,
    UNET_EPOCHS,
    UNET_LR,
    UNET_LR_FACTOR,
    UNET_LR_PATIENCE,
    USE_AMP,
    DEFAULT_THRESHOLD,
)
from src.dataset import get_loaders
from src.losses import CombinedLoss
from src.metrics import compute_all_metrics
from src.models.fcn_baseline import FCNBaseline
from src.models.unet import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_and_config(model_name: str, args):
    """Return (model, epochs, lr, early_stop, lr_patience, lr_factor)."""
    if model_name == "fcn":
        model = FCNBaseline(base_channels=FCN_BASE_CHANNELS)
        epochs     = args.epochs or FCN_EPOCHS
        lr         = args.lr    or FCN_LR
        early_stop = FCN_EARLY_STOP
        patience   = FCN_LR_PATIENCE
        factor     = FCN_LR_FACTOR
    elif model_name == "unet":
        model = UNet(base_channels=UNET_BASE_CHANNELS)
        epochs     = args.epochs or UNET_EPOCHS
        lr         = args.lr    or UNET_LR
        early_stop = UNET_EARLY_STOP
        patience   = UNET_LR_PATIENCE
        factor     = UNET_LR_FACTOR
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose 'fcn' or 'unet'.")
    return model, epochs, lr, early_stop, patience, factor


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: Path, device):
    """Load a checkpoint and return (start_epoch, best_val_loss, history)."""
    print(f"Resuming from checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch    = ckpt.get("epoch", 0) + 1
    best_val_loss  = ckpt.get("best_val_loss", float("inf"))
    history        = ckpt.get("history", {"train_loss": [], "val_loss": [], "val_dice": []})
    return start_epoch, best_val_loss, history


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, history, path: Path):
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_val_loss":   best_val_loss,
            "history":         history,
        },
        path,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="  train", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, threshold=DEFAULT_THRESHOLD):
    model.eval()
    total_loss = 0.0
    all_preds   = []
    all_targets = []

    for images, masks in tqdm(loader, desc="  val  ", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).cpu()
        all_preds.append(probs)
        all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_all_metrics(all_preds, all_targets, threshold=threshold)
    val_loss = total_loss / len(loader.dataset)
    return val_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train a crack segmentation model.")
    parser.add_argument("--model",  required=True, choices=["fcn", "unet"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int,   default=None)
    parser.add_argument("--lr",     type=float, default=None)
    parser.add_argument("--batch",  type=int,   default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, epochs, lr, early_stop, patience, factor = get_model_and_config(args.model, args)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : {args.model.upper()}  ({n_params:,} parameters)")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_loaders(batch_size=args.batch)
    print(f"Train   : {len(train_loader.dataset)} images / {len(train_loader)} batches")
    print(f"Val     : {len(val_loader.dataset)} images / {len(val_loader)} batches")

    # ── Optimiser and scheduler ────────────────────────────────────────────────
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=USE_AMP and device.type == "cuda")
    use_amp = USE_AMP and device.type == "cuda"

    # ── Paths ──────────────────────────────────────────────────────────────────
    resume_ckpt = CHECKPOINT_DIR / f"{args.model}_latest.pth"
    best_ckpt   = CHECKPOINT_DIR / f"{args.model}_best.pth"
    history_path = OUTPUT_DIR / f"{args.model}_history.json"

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    history       = {"train_loss": [], "val_loss": [], "val_dice": []}
    no_improve    = 0

    if args.resume and resume_ckpt.exists():
        start_epoch, best_val_loss, history = load_checkpoint(
            model, optimizer, scheduler, resume_ckpt, device
        )

    print(f"\nStarting training: epochs {start_epoch + 1} -> {epochs}")
    print(f"LR={lr}  batch={args.batch}  AMP={use_amp}  early_stop={early_stop}\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, use_amp
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_metrics["dice"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"dice={val_metrics['dice']:.4f}  iou={val_metrics['iou']:.4f}  "
            f"prec={val_metrics['precision']:.4f}  rec={val_metrics['recall']:.4f}  "
            f"[{elapsed:.0f}s]"
        )

        # ── Checkpoint logic ──────────────────────────────────────────────────
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, history, best_ckpt)
            print(f"  --> New best checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1

        # Always save the latest checkpoint to support resuming.
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, history, resume_ckpt)

        # Save history to JSON after every epoch (for live monitoring).
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= early_stop:
            print(f"\nEarly stopping: no improvement for {early_stop} epochs.")
            break

    print(f"\nTraining complete.")
    print(f"Best checkpoint : {best_ckpt}")
    print(f"History saved   : {history_path}")
    print(f"Best val loss   : {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
