
import json
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
from tqdm import tqdm

# Project paths
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "crack-seg.ndjson"
OUTPUT = ROOT / "data" / "raw"
IMAGE_DIR = OUTPUT / "images"
MASK_DIR = OUTPUT / "masks"

SPLITS = {"train", "val", "test"}
WORKERS = 6

for split in SPLITS:
    (IMAGE_DIR / split).mkdir(parents=True, exist_ok=True)
    (MASK_DIR / split).mkdir(parents=True, exist_ok=True)

if not MANIFEST.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST}\n"
        "Place crack-seg.ndjson in the project root."
    )

# Read image records from the NDJSON manifest
records = []

with MANIFEST.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        if record.get("type") == "image":
            split = record.get("split", "").lower()

            if split not in SPLITS:
                raise ValueError(
                    f"Unexpected split {split!r} for {record.get('file')}"
                )

            records.append(record)

if not records:
    raise RuntimeError("No image records found in the manifest.")

print("Total images in manifest:", len(records))

# Check that the manifest contains all three splits
for split in sorted(SPLITS):
    print(split, ":", sum(r["split"].lower() == split for r in records))


def download_one(record):
    """Download an image and rasterize its polygon annotations."""
    split = record["split"].lower()
    filename = Path(record["file"]).name

    if not filename or filename in {".", ".."}:
        return False, "Invalid filename"

    image_path = IMAGE_DIR / split / filename
    mask_path = MASK_DIR / split / f"{Path(filename).stem}.png"

    # Skip files that are already fully prepared
    if image_path.exists() and mask_path.exists():
        if image_path.stat().st_size > 0 and mask_path.stat().st_size > 0:
            return True, "Already exists"

    url = record.get("url")
    if not url:
        return False, f"{filename}: missing URL"

    # Retry transient network failures
    image = None
    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()

            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Downloaded content is not a valid image")

            break

        except Exception as exc:
            last_error = str(exc)
            time.sleep(attempt + 1)

    if image is None:
        return False, f"{filename}: download failed: {last_error}"

    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        return False, f"{filename}: invalid image dimensions"

    # Black background; annotated crack polygons become white.
    mask = np.zeros((height, width), dtype=np.uint8)
    segments = record.get("annotations", {}).get("segments", [])

    for segment in segments:
        if len(segment) < 7:
            continue

        # YOLO-style polygon: class_id, x1, y1, x2, y2, ...
        coords = np.asarray(segment[1:], dtype=np.float32)

        if len(coords) % 2 != 0:
            continue

        points = coords.reshape(-1, 2)

        # Coordinates are normalized to [0, 1].
        points[:, 0] = np.clip(points[:, 0] * width, 0, width - 1)
        points[:, 1] = np.clip(points[:, 1] * height, 0, height - 1)

        points = np.round(points).astype(np.int32)

        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)

    # Save files only after successfully decoding and preparing them.
    if not cv2.imwrite(str(image_path), image):
        return False, f"{filename}: failed to save image"

    if not cv2.imwrite(str(mask_path), mask):
        image_path.unlink(missing_ok=True)
        return False, f"{filename}: failed to save mask"

    return True, filename


failed = []

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = [executor.submit(download_one, r) for r in records]

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Downloading images and masks"
    ):
        try:
            success, message = future.result()
            if not success:
                failed.append(message)
        except Exception as exc:
            failed.append(str(exc))

print("\nPreparation finished.")
print("Successful or already prepared:", len(records) - len(failed))
print("Failed:", len(failed))

if failed:
    failure_log = ROOT / "outputs" / "download_failures.txt"
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    failure_log.write_text("\n".join(failed), encoding="utf-8")
    print("Failure details saved to:", failure_log)

print("\nDataset directories:")
print("Images:", IMAGE_DIR)
print("Masks :", MASK_DIR)
