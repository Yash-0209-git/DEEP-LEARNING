"""
evaluate.py — Test-set evaluation for Road Crack Detection.

Usage (from project root):

    Evaluate FCN baseline:
        python -m src.evaluate --model fcn

    Evaluate U-Net:
        python -m src.evaluate --model unet

Produces:
    outputs/eval_{model}/
        metrics.json          -- aggregate test metrics (JSON)
        metrics_per_image.csv -- per-image dice, iou, precision, recall
        threshold_sweep.json  -- val-set dice vs threshold (for tuning)
        loss_curve.png        -- training + validation loss over epochs
        metrics_curve.png     -- validation dice over epochs
        predictions.png       -- grid: image | gt mask | pred mask | overlay
        confusion.png         -- pixel-level confusion matrix heatmap
"""

import argparse
import json
import time
from pathlib import Path

import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from src.config import (
    CHECKPOINT_DIR,
    DEFAULT_THRESHOLD,
    FCN_BASE_CHANNELS,
    IMAGE_MEAN,
    IMAGE_STD,
    OUTPUT_DIR,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    UNET_BASE_CHANNELS,
)
from src.dataset import CrackSegmentationDataset, get_loaders
from src.losses import CombinedLoss
from src.metrics import compute_all_metrics, dice_coefficient
from src.models.fcn_baseline import FCNBaseline
from src.models.unet import UNet

plt.style.use("seaborn-v0_8-darkgrid")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(model_name: str, checkpoint_path: Path, device):
    if model_name == "fcn":
        model = FCNBaseline(base_channels=FCN_BASE_CHANNELS)
    else:
        model = UNet(base_channels=UNET_BASE_CHANNELS)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded {model_name.upper()} from {checkpoint_path}")
    return model, ckpt.get("history", {})


def denormalise(tensor):
    """Convert a normalised image tensor [3,H,W] back to uint8 for display."""
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGE_STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ── Threshold sweep on validation set ─────────────────────────────────────────

@torch.no_grad()
def sweep_threshold(model, val_loader, device):
    """Return (best_threshold, sweep_dict) based on max val Dice."""
    print("Sweeping threshold on validation set …")
    all_probs   = []
    all_targets = []

    for images, masks in val_loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_targets.append(masks)

    all_probs   = torch.cat(all_probs,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    thresholds  = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-9, THRESHOLD_STEP)
    sweep       = {}
    best_thresh = DEFAULT_THRESHOLD
    best_dice   = -1.0

    for t in thresholds:
        t = round(float(t), 2)
        m = compute_all_metrics(all_probs, all_targets, threshold=t)
        sweep[str(t)] = m
        if m["dice"] > best_dice:
            best_dice   = m["dice"]
            best_thresh = t

    print(f"Best val threshold: {best_thresh:.2f}  (dice={best_dice:.4f})")
    return best_thresh, sweep


# ── Test-set evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model, test_loader, device, threshold, criterion):
    """Return (aggregate_metrics, per_image_rows, collected_batches)."""
    print("Evaluating on test set …")
    all_probs   = []
    all_targets = []
    all_images  = []   # for visualisation (keep first 3 batches only)
    total_loss  = 0.0
    timing      = []

    for i, (images, masks) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        t0 = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        elapsed = time.perf_counter() - t0
        timing.append(elapsed / images.size(0))   # per-image time

        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_targets.append(masks.cpu())
        if i < 3:
            all_images.append(images.cpu())

    all_probs   = torch.cat(all_probs,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Aggregate metrics
    agg = compute_all_metrics(all_probs, all_targets, threshold=threshold)
    agg["test_loss"]            = total_loss / len(test_loader.dataset)
    agg["mean_inference_ms"]    = round(float(np.mean(timing)) * 1000, 2)
    agg["threshold_used"]       = threshold

    # Per-image metrics
    rows = []
    ds   = test_loader.dataset
    for idx in range(len(all_probs)):
        fname = ds.image_paths[idx].name
        m = compute_all_metrics(all_probs[idx:idx+1], all_targets[idx:idx+1], threshold=threshold)
        rows.append({"file": fname, **m})

    collected = {
        "images": torch.cat(all_images, dim=0)[:24],
        "probs":  all_probs[:24],
        "masks":  all_targets[:24],
    }
    return agg, rows, collected, timing


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, out_dir: Path, model_name: str):
    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss",   [])
    val_dice   = history.get("val_dice",   [])

    if not train_loss:
        print("No history found — skipping loss/metric curves.")
        return

    epochs = range(1, len(train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train loss", color="#2196F3")
    axes[0].plot(epochs, val_loss,   label="Val loss",   color="#FF5722")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name.upper()} — Loss curves")
    axes[0].legend(); axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[1].plot(epochs, val_dice, label="Val Dice", color="#4CAF50")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[1].set_title(f"{model_name.upper()} — Validation Dice")
    axes[1].legend(); axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = out_dir / "loss_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved: {path}")


def plot_predictions(collected: dict, out_dir: Path, model_name: str, threshold: float, n=8):
    images = collected["images"][:n]
    probs  = collected["probs"][:n]
    masks  = collected["masks"][:n]

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis]

    for i in range(n):
        img  = denormalise(images[i])
        gt   = (masks[i, 0].numpy() * 255).astype(np.uint8)
        pred = ((probs[i, 0].numpy() >= threshold) * 255).astype(np.uint8)

        # Overlay: predicted cracks in red
        overlay = img.copy()
        crack_px = pred > 0
        overlay[crack_px] = (0.5 * overlay[crack_px] + 0.5 * np.array([255, 60, 60])).astype(np.uint8)

        axes[i, 0].imshow(img); axes[i, 0].set_title("Input image")
        axes[i, 1].imshow(gt,   cmap="gray", vmin=0, vmax=255); axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=255); axes[i, 2].set_title("Prediction")
        axes[i, 3].imshow(overlay); axes[i, 3].set_title("Overlay")

        d = dice_coefficient(pred // 255, gt // 255)
        for ax in axes[i]:
            ax.axis("off")
        axes[i, 2].set_title(f"Pred  (Dice={d:.3f})")

    plt.suptitle(f"{model_name.upper()} — Test set predictions (threshold={threshold:.2f})", y=1.01)
    plt.tight_layout()
    path = out_dir / "predictions.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Predictions grid saved: {path}")


def plot_confusion(agg: dict, out_dir: Path, model_name: str):
    """Simple 2x2 pixel confusion matrix from aggregate metrics."""
    # Reconstruct TP/FP/TN/FN from precision/recall/dice
    # (exact pixel counts unavailable from aggregated metrics, so we label with rates)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array([
        [agg.get("pixel_acc", 0), 1 - agg.get("precision", 0)],
        [1 - agg.get("recall", 0), agg.get("precision", 0) * agg.get("recall", 0)]
    ])
    labels = [["True Neg\n(background\ncorrect)",  "False Pos\n(false alarm)"],
              ["False Neg\n(missed crack)", "True Pos\n(crack correct)"]]

    im = ax.imshow([[agg.get("pixel_acc", 0), 1-agg.get("precision", 0)],
                    [1-agg.get("recall", 0), agg.get("recall", 0)]],
                   cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    vals = [[f"TN\nacc={agg.get('pixel_acc',0):.3f}",
             f"FP\n(1-prec)={1-agg.get('precision',0):.3f}"],
            [f"FN\n(1-rec)={1-agg.get('recall',0):.3f}",
             f"TP\nrec={agg.get('recall',0):.3f}"]]
    for r in range(2):
        for c in range(2):
            ax.text(c, r, vals[r][c], ha="center", va="center", fontsize=9,
                    color="black" if cm[r, c] < 0.6 else "white")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: BG", "Pred: Crack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True: BG", "True: Crack"])
    ax.set_title(f"{model_name.upper()} — Pixel confusion (rate-normalised)")
    plt.tight_layout()
    path = out_dir / "confusion.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {path}")


def plot_threshold_sweep(sweep: dict, best_thresh: float, out_dir: Path, model_name: str):
    thresholds = sorted(float(k) for k in sweep)
    dices      = [sweep[str(round(t, 2))]["dice"]  for t in thresholds]
    ious       = [sweep[str(round(t, 2))]["iou"]   for t in thresholds]
    precs      = [sweep[str(round(t, 2))]["precision"] for t in thresholds]
    recs       = [sweep[str(round(t, 2))]["recall"]    for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, dices, label="Dice",      color="#4CAF50", linewidth=2)
    ax.plot(thresholds, ious,  label="IoU",       color="#2196F3")
    ax.plot(thresholds, precs, label="Precision", color="#FF9800", linestyle="--")
    ax.plot(thresholds, recs,  label="Recall",    color="#E91E63",  linestyle="--")
    ax.axvline(best_thresh, color="red", linestyle=":", label=f"Best t={best_thresh:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"{model_name.upper()} — Val-set metric vs threshold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = out_dir / "threshold_sweep.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Threshold sweep plot saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["fcn", "unet"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = OUTPUT_DIR / f"eval_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = CHECKPOINT_DIR / f"{args.model}_best.pth"
    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {best_ckpt}. "
            f"Train first: python -m src.train --model {args.model}"
        )

    # ── Load model ─────────────────────────────────────────────────────────────
    model, history = load_model(args.model, best_ckpt, device)
    criterion = CombinedLoss()

    # ── Data loaders ────────────────────────────────────────────────────────────
    _, val_loader, test_loader = get_loaders()

    # ── Threshold sweep ─────────────────────────────────────────────────────────
    best_thresh, sweep = sweep_threshold(model, val_loader, device)

    # ── Test evaluation ─────────────────────────────────────────────────────────
    agg, per_image, collected, timing = evaluate_test(
        model, test_loader, device, best_thresh, criterion
    )

    # ── Print results ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {args.model.upper()} — Test Set Results (threshold={best_thresh:.2f})")
    print(f"{'='*55}")
    print(f"  Dice coefficient  : {agg['dice']:.4f}")
    print(f"  IoU               : {agg['iou']:.4f}")
    print(f"  Precision         : {agg['precision']:.4f}")
    print(f"  Recall            : {agg['recall']:.4f}")
    print(f"  Pixel accuracy    : {agg['pixel_acc']:.4f}  (NOTE: misleading metric)")
    print(f"  Test loss         : {agg['test_loss']:.4f}")
    print(f"  Mean inference    : {agg['mean_inference_ms']:.1f} ms/image")
    print(f"{'='*55}\n")

    # ── Save JSON metrics ───────────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"Metrics saved: {metrics_path}")

    # ── Save per-image CSV ──────────────────────────────────────────────────────
    csv_path = out_dir / "metrics_per_image.csv"
    if per_image:
        fieldnames = list(per_image[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image)
        print(f"Per-image metrics saved: {csv_path}")

    # ── Save threshold sweep ────────────────────────────────────────────────────
    sweep_path = out_dir / "threshold_sweep.json"
    sweep_path.write_text(json.dumps({"best_threshold": best_thresh, "sweep": sweep}, indent=2), encoding="utf-8")

    # ── Plots ───────────────────────────────────────────────────────────────────
    plot_loss_curves(history, out_dir, args.model)
    plot_threshold_sweep(sweep, best_thresh, out_dir, args.model)
    plot_predictions(collected, out_dir, args.model, best_thresh, n=min(8, len(collected["images"])))
    plot_confusion(agg, out_dir, args.model)

    print(f"\nAll evaluation outputs in: {out_dir}")


if __name__ == "__main__":
    main()
