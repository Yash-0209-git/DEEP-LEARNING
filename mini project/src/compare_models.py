"""
compare_models.py — Side-by-side comparison of FCN baseline and U-Net results.

Usage (after both models are evaluated):
    python -m src.compare_models

Reads outputs/eval_fcn/metrics.json and outputs/eval_unet/metrics.json.
Produces outputs/comparison_table.md and outputs/comparison_bar.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

METRICS_TO_COMPARE = ["dice", "iou", "precision", "recall", "pixel_acc"]
METRIC_LABELS      = {
    "dice":      "Dice",
    "iou":       "IoU",
    "precision": "Precision",
    "recall":    "Recall",
    "pixel_acc": "Pixel Acc.",
}


def load_metrics(model_name: str) -> dict | None:
    path = OUTPUT_DIR / f"eval_{model_name}" / "metrics.json"
    if not path.exists():
        print(f"WARNING: {path} not found — run: python -m src.evaluate --model {model_name}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    fcn_m  = load_metrics("fcn")
    unet_m = load_metrics("unet")

    if not fcn_m and not unet_m:
        print("No evaluation results found. Evaluate models first.")
        return

    # ── Markdown comparison table ──────────────────────────────────────────────
    lines = ["# Model Comparison — Road Crack Detection\n"]
    lines.append("| Metric | FCN Baseline | U-Net | Improvement |")
    lines.append("|--------|-------------|-------|-------------|")

    for key in METRICS_TO_COMPARE:
        label    = METRIC_LABELS.get(key, key)
        fcn_val  = fcn_m.get(key, float("nan")) if fcn_m  else float("nan")
        unet_val = unet_m.get(key, float("nan")) if unet_m else float("nan")

        if not np.isnan(fcn_val) and not np.isnan(unet_val):
            diff = unet_val - fcn_val
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        else:
            diff_str = "N/A"

        fcn_str  = f"{fcn_val:.4f}"  if not np.isnan(fcn_val)  else "N/A"
        unet_str = f"{unet_val:.4f}" if not np.isnan(unet_val) else "N/A"
        lines.append(f"| {label} | {fcn_str} | {unet_str} | {diff_str} |")

    # Extra rows
    for key, label in [("test_loss", "Test loss"), ("mean_inference_ms", "Inference (ms)")]:
        fcn_val  = fcn_m.get(key, float("nan")) if fcn_m  else float("nan")
        unet_val = unet_m.get(key, float("nan")) if unet_m else float("nan")
        fcn_str  = f"{fcn_val:.2f}"  if not np.isnan(fcn_val)  else "N/A"
        unet_str = f"{unet_val:.2f}" if not np.isnan(unet_val) else "N/A"
        lines.append(f"| {label} | {fcn_str} | {unet_str} | — |")

    table_path = OUTPUT_DIR / "comparison_table.md"
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison table saved: {table_path}")
    print("\n".join(lines))

    # ── Bar chart ──────────────────────────────────────────────────────────────
    models   = []
    datasets = {k: [] for k in METRICS_TO_COMPARE}

    for label, m in [("FCN Baseline", fcn_m), ("U-Net", unet_m)]:
        if m:
            models.append(label)
            for k in METRICS_TO_COMPARE:
                datasets[k].append(m.get(k, 0))

    if len(models) < 1:
        return

    x     = np.arange(len(models))
    n_met = len(METRICS_TO_COMPARE)
    width = 0.8 / n_met

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"]

    for i, key in enumerate(METRICS_TO_COMPARE):
        offset = (i - n_met / 2 + 0.5) * width
        bars = ax.bar(x + offset, datasets[key], width, label=METRIC_LABELS[key], color=colours[i])
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("FCN Baseline vs U-Net — Test Set Metrics")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    bar_path = OUTPUT_DIR / "comparison_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved: {bar_path}")


if __name__ == "__main__":
    main()
