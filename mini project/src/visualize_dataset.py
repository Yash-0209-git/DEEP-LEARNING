
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "raw"

SPLIT = "train"
NUM_SAMPLES = 6

image_dir = BASE / "images" / SPLIT
mask_dir = BASE / "masks" / SPLIT

image_paths = sorted(
    p for p in image_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

valid_pairs = [
    p for p in image_paths
    if (mask_dir / f"{p.stem}.png").exists()
]

if not valid_pairs:
    raise RuntimeError("No matching image-mask pairs found.")

random.seed(42)
samples = random.sample(valid_pairs, min(NUM_SAMPLES, len(valid_pairs)))

fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))

if len(samples) == 1:
    axes = np.expand_dims(axes, axis=0)

for row, image_path in enumerate(samples):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_path = mask_dir / f"{image_path.stem}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Could not read:", image_path.name)
        continue

    mask_binary = mask > 0

    # Overlay the annotated crack regions in red.
    overlay = image.copy()
    overlay[mask_binary] = (
        0.6 * overlay[mask_binary] + 0.4 * np.array([255, 0, 0])
    ).astype(np.uint8)

    axes[row, 0].imshow(image)
    axes[row, 0].set_title("Original image")

    axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[row, 1].set_title("Ground-truth mask")

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title("Crack overlay")

    for ax in axes[row]:
        ax.axis("off")

    print(
        image_path.name,
        "| crack pixels:",
        round(mask_binary.mean() * 100, 2), "%"
    )

plt.tight_layout()

output_dir = ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "dataset_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved visualization to:", output_path)
