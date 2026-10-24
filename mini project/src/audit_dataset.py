"""
audit_dataset.py — Phase One: Comprehensive dataset audit for Road Crack Detection.

Checks performed:
  1. Image and mask counts per split.
  2. Every image has a paired mask (same stem).
  3. Image and mask dimensions (all should be 416 x 416).
  4. Unreadable / corrupt files.
  5. Mask pixel-value sanity (only 0 and 255 expected).
  6. Empty masks (100% background).
  7. Suspiciously over-annotated masks (>= 50% crack pixels).
  8. Per-split crack-pixel statistics (mean / std / min / max %).
  9. Filename-level duplicate detection across splits (potential data leakage).
 10. Saves JSON report to  outputs/audit/audit_results.json
     Saves markdown report to  outputs/audit/audit_report.md
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUTPUT = ROOT / "outputs" / "audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Thresholds for flagging masks
EMPTY_MASK_THRESHOLD = 0.0          # 0 % crack → empty mask
HIGH_COVERAGE_THRESHOLD = 50.0      # >= 50 % crack → suspiciously over-annotated

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Helpers ────────────────────────────────────────────────────────────────────

def collect_files(directory: Path, extensions: set) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in extensions)


def read_image_safe(path: Path):
    """Return (array, error_string). error_string is None on success."""
    try:
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            return None, "cv2.imread returned None"
        return arr, None
    except Exception as exc:
        return None, str(exc)


def read_mask_safe(path: Path):
    try:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return None, "cv2.imread returned None"
        return arr, None
    except Exception as exc:
        return None, str(exc)


# ── Per-split audit ────────────────────────────────────────────────────────────

def audit_split(split: str) -> dict:
    image_dir = DATA / "images" / split
    mask_dir  = DATA / "masks"  / split

    if not image_dir.exists():
        return {"split": split, "error": f"Image directory not found: {image_dir}"}
    if not mask_dir.exists():
        return {"split": split, "error": f"Mask directory not found: {mask_dir}"}

    image_files = collect_files(image_dir, IMAGE_EXTENSIONS)
    mask_files  = collect_files(mask_dir,  {".png"})

    mask_stem_set = {p.stem for p in mask_files}

    # ── Pairing ──────────────────────────────────────────────────────────────
    images_missing_mask = [p.name for p in image_files if p.stem not in mask_stem_set]
    image_stem_set = {p.stem for p in image_files}
    masks_missing_image = [p.name for p in mask_files if p.stem not in image_stem_set]

    # ── Per-file checks ───────────────────────────────────────────────────────
    corrupt_images  = []
    corrupt_masks   = []
    dim_mismatch    = []      # image and mask spatial dims differ
    wrong_img_dims  = []      # not 416 x 416
    wrong_mask_dims = []
    unexpected_mask_values = []
    empty_masks     = []
    high_coverage   = []

    crack_percentages = []   # for all valid paired masks

    paired_images = [p for p in image_files if p.stem in mask_stem_set]

    for img_path in paired_images:
        mask_path = mask_dir / f"{img_path.stem}.png"

        img, img_err = read_image_safe(img_path)
        msk, msk_err = read_mask_safe(mask_path)

        if img_err:
            corrupt_images.append({"file": img_path.name, "error": img_err})
            continue

        if msk_err:
            corrupt_masks.append({"file": mask_path.name, "error": msk_err})
            continue

        ih, iw = img.shape[:2]
        mh, mw = msk.shape[:2]

        if ih != 416 or iw != 416:
            wrong_img_dims.append({"file": img_path.name, "shape": f"{ih}x{iw}"})

        if mh != 416 or mw != 416:
            wrong_mask_dims.append({"file": mask_path.name, "shape": f"{mh}x{mw}"})

        if ih != mh or iw != mw:
            dim_mismatch.append({
                "image": img_path.name,
                "image_shape": f"{ih}x{iw}",
                "mask_shape": f"{mh}x{mw}",
            })

        unique_vals = set(np.unique(msk).tolist())
        expected    = {0, 255}
        unexpected  = unique_vals - expected
        if unexpected:
            unexpected_mask_values.append({
                "file": mask_path.name,
                "unexpected_values": sorted(unexpected),
            })

        crack_pct = float(np.mean(msk > 0) * 100)
        crack_percentages.append(crack_pct)

        if crack_pct <= EMPTY_MASK_THRESHOLD:
            empty_masks.append({"file": mask_path.name, "crack_pct": crack_pct})

        if crack_pct >= HIGH_COVERAGE_THRESHOLD:
            high_coverage.append({"file": mask_path.name, "crack_pct": round(crack_pct, 2)})

    # ── Aggregate crack statistics ────────────────────────────────────────────
    if crack_percentages:
        crack_stats = {
            "mean_pct":   round(float(np.mean(crack_percentages)),  3),
            "std_pct":    round(float(np.std(crack_percentages)),   3),
            "min_pct":    round(float(np.min(crack_percentages)),   3),
            "max_pct":    round(float(np.max(crack_percentages)),   3),
            "median_pct": round(float(np.median(crack_percentages)),3),
            "pct_empty":  round(100 * sum(1 for x in crack_percentages if x <= EMPTY_MASK_THRESHOLD) / len(crack_percentages), 2),
        }
    else:
        crack_stats = {}

    return {
        "split": split,
        "total_images": len(image_files),
        "total_masks":  len(mask_files),
        "paired_images": len(paired_images),
        "images_missing_mask": images_missing_mask,
        "masks_missing_image": masks_missing_image,
        "corrupt_images":  corrupt_images,
        "corrupt_masks":   corrupt_masks,
        "dim_mismatch":    dim_mismatch,
        "wrong_img_dims":  wrong_img_dims,
        "wrong_mask_dims": wrong_mask_dims,
        "unexpected_mask_values": unexpected_mask_values,
        "empty_masks":     empty_masks,
        "high_coverage_masks": high_coverage,
        "crack_stats":     crack_stats,
    }


# ── Cross-split duplicate check ───────────────────────────────────────────────

def check_cross_split_duplicates() -> dict:
    """
    Detect filenames (by stem) that appear in more than one split.
    This is a proxy for data leakage — it cannot guarantee pixel-level
    uniqueness without hashing every image, but filename overlap is a
    strong signal because stems encode original Roboflow record IDs.
    """
    stem_to_splits: dict[str, list[str]] = {}

    for split in SPLITS:
        image_dir = DATA / "images" / split
        if not image_dir.exists():
            continue
        for p in collect_files(image_dir, IMAGE_EXTENSIONS):
            stem_to_splits.setdefault(p.stem, []).append(split)

    duplicates = {
        stem: splits
        for stem, splits in stem_to_splits.items()
        if len(splits) > 1
    }

    return {
        "total_stems_checked": len(stem_to_splits),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
    }


# ── Run audit ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Road Crack Detection — Dataset Audit")
print("=" * 60)

split_results = {}
for split in SPLITS:
    print(f"\n[{split.upper()}] Auditing …", flush=True)
    result = audit_split(split)
    split_results[split] = result

    print(f"  Images          : {result.get('total_images', 'N/A')}")
    print(f"  Masks           : {result.get('total_masks',  'N/A')}")
    print(f"  Paired          : {result.get('paired_images','N/A')}")
    print(f"  Missing masks   : {len(result.get('images_missing_mask', []))}")
    print(f"  Missing images  : {len(result.get('masks_missing_image', []))}")
    print(f"  Corrupt images  : {len(result.get('corrupt_images',  []))}")
    print(f"  Corrupt masks   : {len(result.get('corrupt_masks',   []))}")
    print(f"  Dim mismatches  : {len(result.get('dim_mismatch',    []))}")
    print(f"  Wrong img dims  : {len(result.get('wrong_img_dims',  []))}")
    print(f"  Wrong mask dims : {len(result.get('wrong_mask_dims', []))}")
    print(f"  Unexpected vals : {len(result.get('unexpected_mask_values', []))}")
    print(f"  Empty masks (0%): {len(result.get('empty_masks',    []))}")
    print(f"  High-cov masks  : {len(result.get('high_coverage_masks', []))}")

    cs = result.get("crack_stats", {})
    if cs:
        print(f"  Crack % mean    : {cs['mean_pct']:.3f}%")
        print(f"  Crack % std     : {cs['std_pct']:.3f}%")
        print(f"  Crack % min     : {cs['min_pct']:.3f}%")
        print(f"  Crack % max     : {cs['max_pct']:.3f}%")
        print(f"  Crack % median  : {cs['median_pct']:.3f}%")
        print(f"  Fully-empty %   : {cs['pct_empty']:.2f}% of masks")

print("\n[CROSS-SPLIT] Checking filename duplicates …", flush=True)
dup_result = check_cross_split_duplicates()
print(f"  Stems checked   : {dup_result['total_stems_checked']}")
print(f"  Duplicates found: {dup_result['duplicate_count']}")
if dup_result["duplicates"]:
    print("  WARNING: Filename overlap detected between splits (potential leakage)!")
    for stem, splits in list(dup_result["duplicates"].items())[:10]:
        print(f"    {stem} -> {splits}")
else:
    print("  No filename overlap between splits. [OK]")

# ── Save JSON report ──────────────────────────────────────────────────────────
audit_json = {
    "splits": split_results,
    "cross_split_duplicates": dup_result,
}

json_path = OUTPUT / "audit_results.json"
json_path.write_text(json.dumps(audit_json, indent=2), encoding="utf-8")
print(f"\nJSON report saved: {json_path}")

# ── Save Markdown report ──────────────────────────────────────────────────────
lines = ["# Dataset Audit Report -- Road Crack Detection\n"]

for split in SPLITS:
    r = split_results[split]
    cs = r.get("crack_stats", {})
    lines.append(f"## Split: `{split}`\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total images | {r.get('total_images','N/A')} |")
    lines.append(f"| Total masks | {r.get('total_masks','N/A')} |")
    lines.append(f"| Paired | {r.get('paired_images','N/A')} |")
    lines.append(f"| Images missing mask | {len(r.get('images_missing_mask',[]))} |")
    lines.append(f"| Masks missing image | {len(r.get('masks_missing_image',[]))} |")
    lines.append(f"| Corrupt images | {len(r.get('corrupt_images',[]))} |")
    lines.append(f"| Corrupt masks | {len(r.get('corrupt_masks',[]))} |")
    lines.append(f"| Dimension mismatches | {len(r.get('dim_mismatch',[]))} |")
    lines.append(f"| Unexpected mask values | {len(r.get('unexpected_mask_values',[]))} |")
    lines.append(f"| Empty masks (0 % crack) | {len(r.get('empty_masks',[]))} |")
    lines.append(f"| High-coverage masks (>=50 %) | {len(r.get('high_coverage_masks',[]))} |")
    if cs:
        lines.append(f"| Mean crack % | {cs['mean_pct']} % |")
        lines.append(f"| Std crack % | {cs['std_pct']} % |")
        lines.append(f"| Min crack % | {cs['min_pct']} % |")
        lines.append(f"| Max crack % | {cs['max_pct']} % |")
        lines.append(f"| Median crack % | {cs['median_pct']} % |")
        lines.append(f"| Fully-empty masks | {cs['pct_empty']} % of split |")
    lines.append("")

lines.append("## Cross-Split Duplicate Check\n")
lines.append(f"- Total stems checked: {dup_result['total_stems_checked']}")
lines.append(f"- Duplicates found: {dup_result['duplicate_count']}")
if dup_result["duplicates"]:
    lines.append("\n> WARNING: Filename overlap between splits detected.")
    lines.append("\n| Stem | Splits |")
    lines.append("|------|--------|")
    for stem, splits in list(dup_result["duplicates"].items())[:20]:
        lines.append(f"| `{stem}` | {', '.join(splits)} |")
else:
    lines.append("\nNo filename overlap between splits detected. [OK]")

md_path = OUTPUT / "audit_report.md"
md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Markdown report saved: {md_path}")
print("\nAudit complete.")

