
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "raw"

for split in ("train", "val", "test"):
    images = BASE / "images" / split
    masks = BASE / "masks" / split

    image_files = sorted(
        p for p in images.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    mask_files = sorted(masks.glob("*.png"))
    mask_names = {p.stem for p in mask_files}

    missing_masks = [
        p.name for p in image_files
        if p.stem not in mask_names
    ]

    print(f"\n{split.upper()}")
    print("Images:", len(image_files))
    print("Masks:", len(mask_files))
    print("Images missing masks:", len(missing_masks))

    if image_files:
        img = cv2.imread(str(image_files[0]))
        mask_path = masks / f"{image_files[0].stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            print("Sample image shape:", img.shape)

        if mask is not None:
            print("Sample mask shape:", mask.shape)
            print("Mask pixel values:", np.unique(mask)[:10])
            print("Crack pixel percentage:",
                  round(np.mean(mask > 0) * 100, 3), "%")
