"""Convert TTPLA polygon annotations to binary mask PNGs.

TTPLA (Abdelfattah et al., ACCV 2020) ships polygon annotations as
JSON files. Our trainer expects binary mask PNGs alongside images. This
script bridges the two formats.

Usage
-----
    python -m scripts.ttpla_to_masks \\
        --root /path/to/ttpla \\
        --classes cable wire conductor \\
        --out /path/to/ttpla/masks

The script is tolerant of TTPLA-version drift: it auto-detects the
annotation format (COCO-style, LabelMe-style, or per-image JSON) by
sampling the first annotation file. If detection fails, override with
``--format`` explicitly.

Limitations
-----------
- Only polygon-style annotations are converted; bbox-only datasets
  cannot produce per-pixel masks without further work.
- Multi-class semantic masks are not supported here — we collapse all
  named classes onto a single binary "conductor" channel. This matches
  the current trainer's expectation.

For production multi-class training, write a separate conversion that
emits PNG-with-class-indices and update the dataset class accordingly.

Reference
---------
TTPLA dataset: Abdelfattah, R., Wang, X., & Wang, S. (2020). TTPLA: An
Aerial-Image Dataset for Detection and Segmentation of Transmission
Towers and Power Lines. ACCV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Class names that map to "conductor" by default. Override via --classes.
DEFAULT_CONDUCTOR_CLASSES = {
    "cable",
    "wire",
    "conductor",
    "powerline",
    "power_line",
    "line",
}


def detect_format(sample: dict) -> str:
    """Sniff the JSON structure of an annotation file."""
    if "shapes" in sample and isinstance(sample["shapes"], list):
        return "labelme"
    if "annotations" in sample and "images" in sample:
        return "coco"
    if "objects" in sample:
        return "ttpla_native"  # TTPLA's own simple per-image format
    return "unknown"


def find_image(images_dir: Path, basename: str) -> Path | None:
    """Look up an image by stem, tolerating common extensions."""
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        p = images_dir / f"{basename}{ext}"
        if p.exists():
            return p
    return None


def rasterise_labelme(
    annotation: dict,
    image_size: tuple[int, int],
    classes: set[str],
) -> np.ndarray:
    """LabelMe-style: top-level 'shapes' list with 'label' and 'points'."""
    h, w = image_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for shape in annotation.get("shapes", []):
        label = str(shape.get("label", "")).lower().strip()
        if label not in classes:
            continue
        pts = [(float(x), float(y)) for x, y in shape.get("points", [])]
        if len(pts) < 2:
            continue
        if shape.get("shape_type", "polygon") == "line":
            draw.line(pts, fill=255, width=3)
        else:
            draw.polygon(pts, fill=255)
    return np.array(mask, dtype=np.uint8)


def rasterise_ttpla_native(
    annotation: dict,
    image_size: tuple[int, int],
    classes: set[str],
) -> np.ndarray:
    """TTPLA-native: 'objects' list with 'classTitle' and 'points'.

    Adjust this function to match the actual TTPLA release format if
    different — verify by reading a real annotation file.
    """
    h, w = image_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for obj in annotation.get("objects", []):
        title = str(obj.get("classTitle", obj.get("class", ""))).lower().strip()
        if title not in classes:
            continue
        # Try common point-storage shapes; adjust as the real format reveals
        points = obj.get("points", {}).get("exterior", obj.get("points", []))
        if not points:
            continue
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 2:
            continue
        draw.polygon(pts, fill=255) if len(pts) >= 3 else draw.line(pts, fill=255, width=3)
    return np.array(mask, dtype=np.uint8)


def rasterise_coco(
    annotation: dict,
    image_size: tuple[int, int],
    classes: set[str],
    image_id: int,
) -> np.ndarray:
    """COCO-style: top-level dict with 'images', 'annotations', 'categories'."""
    h, w = image_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    cats = {c["id"]: c["name"].lower().strip() for c in annotation.get("categories", [])}
    target_ids = {cid for cid, name in cats.items() if name in classes}
    for ann in annotation.get("annotations", []):
        if ann.get("image_id") != image_id:
            continue
        if ann.get("category_id") not in target_ids:
            continue
        for seg in ann.get("segmentation", []):
            if not seg or len(seg) < 6:
                continue
            pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            draw.polygon(pts, fill=255)
    return np.array(mask, dtype=np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, required=True, help="TTPLA dataset root")
    p.add_argument(
        "--annotations-dir",
        type=Path,
        default=None,
        help="Override location of annotation files (default: <root>/annotations)",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override location of images (default: <root>/images)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output mask directory (default: <root>/masks)",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=sorted(DEFAULT_CONDUCTOR_CLASSES),
        help="Class names to treat as conductor (case-insensitive)",
    )
    p.add_argument(
        "--format",
        choices=["auto", "labelme", "coco", "ttpla_native"],
        default="auto",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate without writing files")
    p.add_argument("--limit", type=int, default=0, help="Process only N files (0 = all)")
    args = p.parse_args()

    root: Path = args.root
    images_dir: Path = args.images_dir or (root / "images")
    annotations_dir: Path = args.annotations_dir or (root / "annotations")
    out_dir: Path = args.out or (root / "masks")
    classes = {c.lower().strip() for c in args.classes}

    if not images_dir.is_dir():
        print(f"ERROR: images dir not found: {images_dir}", file=sys.stderr)
        return 2
    if not annotations_dir.is_dir():
        print(f"ERROR: annotations dir not found: {annotations_dir}", file=sys.stderr)
        return 2
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # COCO datasets typically have one big JSON file; everything else is
    # one annotation file per image.
    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: no JSON files in {annotations_dir}", file=sys.stderr)
        return 2

    fmt = args.format
    if fmt == "auto":
        sample = json.loads(json_files[0].read_text())
        fmt = detect_format(sample)
        print(f"Detected annotation format: {fmt}")
        if fmt == "unknown":
            print("Could not detect format; pass --format explicitly", file=sys.stderr)
            return 3

    n_processed, n_pos_pixels, n_total_pixels, n_skipped = 0, 0, 0, 0

    if fmt == "coco":
        # One JSON, many images
        coco = json.loads(json_files[0].read_text())
        coco_images = coco.get("images", [])
        if args.limit:
            coco_images = coco_images[: args.limit]
        for img_meta in coco_images:
            stem = Path(img_meta["file_name"]).stem
            img_path = find_image(images_dir, stem)
            if img_path is None:
                print(f"  warn: missing image for {stem}", file=sys.stderr)
                n_skipped += 1
                continue
            with Image.open(img_path) as im:
                size = im.size  # (w, h)
            mask = rasterise_coco(coco, (size[1], size[0]), classes, img_meta["id"])
            n_processed += 1
            n_pos_pixels += int((mask > 0).sum())
            n_total_pixels += mask.size
            if not args.dry_run:
                Image.fromarray(mask, mode="L").save(out_dir / f"{stem}.png")
    else:
        # One JSON per image (labelme or ttpla_native)
        rasterise = (
            rasterise_labelme if fmt == "labelme" else rasterise_ttpla_native
        )
        if args.limit:
            json_files = json_files[: args.limit]
        for jf in json_files:
            ann = json.loads(jf.read_text())
            stem = jf.stem
            img_path = find_image(images_dir, stem)
            if img_path is None:
                print(f"  warn: missing image for {stem}", file=sys.stderr)
                n_skipped += 1
                continue
            with Image.open(img_path) as im:
                size = im.size  # (w, h)
            mask = rasterise(ann, (size[1], size[0]), classes)
            n_processed += 1
            n_pos_pixels += int((mask > 0).sum())
            n_total_pixels += mask.size
            if not args.dry_run:
                Image.fromarray(mask, mode="L").save(out_dir / f"{stem}.png")

    pct_pos = 100.0 * n_pos_pixels / max(n_total_pixels, 1)
    print(
        f"\nProcessed {n_processed} images "
        f"({n_skipped} skipped). "
        f"Conductor pixel fraction: {pct_pos:.3f}% "
        f"({n_pos_pixels:,} / {n_total_pixels:,})."
    )
    if pct_pos < 0.05:
        print(
            "WARNING: conductor pixel fraction is very low. "
            "Verify class names match the dataset.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
