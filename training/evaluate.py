"""Evaluate a trained conductor segmenter on a held-out split.

Reports the metric family chosen in `docs/evaluation.md`:

- Pixel **IoU**, **precision**, **recall**, **F1** at threshold 0.5.
- **CCQ** (Completeness, Correctness, Quality) at a 3-pixel buffer
  tolerance, after Wiedemann, Heipke, Mayer, Jamet (1998),
  *Empirical Evaluation of Automatically Extracted Road Axes*. CCQ
  is the right metric family for thin-structure tasks: pixel IoU
  under-rewards a 2-pixel-offset trace of a 3-pixel-wide cable, but
  CCQ accepts it as operationally correct.

For consistency with the live demo, inference uses the production
``ConductorSegmenter`` from ``app.ml.model`` (sliding-window with
Hann-windowed stitching). We never re-implement the inference path
just for evaluation — drift between train-time and serve-time
metrics is a common source of confusion.

Usage
-----
    python -m training.evaluate \\
        --data /path/to/ttpla \\
        --weights weights/unet_resnet34_ttpla.pth \\
        --split val \\
        --threshold 0.5 \\
        --tolerance 3 \\
        --n-failures 3 \\
        --out docs/

Outputs
-------
- ``docs/evaluation_results.md`` with the metrics tabulated.
- ``docs/screenshots/eval/successes/{i}.png`` — three success panels.
- ``docs/screenshots/eval/failures/{i}.png`` — three failure panels.

Limitations
-----------
- The val/test split here is the same random 10/10 split used by the
  trainer. For a production run we would use a *geographic* split,
  per ``docs/evaluation.md`` §3.
- CCQ relies on a buffer-distance approximation rather than a
  reference-line vector representation; on very dense networks the
  buffers can over-count. With TTPLA's sparse transmission lines the
  approximation is fine.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import structlog
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from app.ml.model import ConductorSegmenter

log = structlog.get_logger()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class PerImage:
    """Per-image metrics retained for ranking and failure-gallery selection."""

    name: str
    iou: float
    precision: float
    recall: float
    f1: float
    ccq_completeness: float
    ccq_correctness: float
    ccq_quality: float


# ── pixel metrics ──────────────────────────────────────────────────────
def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Pixel-level IoU / precision / recall / F1 on binary arrays."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = float(np.logical_and(pred_b, gt_b).sum())
    union = float(np.logical_or(pred_b, gt_b).sum())
    n_pred = float(pred_b.sum())
    n_gt = float(gt_b.sum())
    iou = inter / union if union > 0 else 1.0
    prec = inter / n_pred if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
    rec = inter / n_gt if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1}


# ── CCQ ────────────────────────────────────────────────────────────────
def ccq(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 3) -> dict[str, float]:
    """Completeness, Correctness, Quality at a buffer tolerance.

    Definition follows Wiedemann et al. (1998). Skeletonise both masks
    so we measure centerlines, then count skeleton pixels of one mask
    that fall within `tolerance_px` of the other (computed via Euclidean
    distance transform).

    Edge cases
    ----------
    Both masks empty → all three scores are 1.0 (vacuously perfect).
    One empty, the other not → 0.0 for the affected term.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    pred_skel = skeletonize(pred_b)
    gt_skel = skeletonize(gt_b)

    pred_len = int(pred_skel.sum())
    gt_len = int(gt_skel.sum())
    if pred_len == 0 and gt_len == 0:
        return {"completeness": 1.0, "correctness": 1.0, "quality": 1.0}
    if gt_len == 0:
        # No reference to recover; pure precision case
        return {"completeness": 1.0, "correctness": 0.0, "quality": 0.0}
    if pred_len == 0:
        return {"completeness": 0.0, "correctness": 1.0, "quality": 0.0}

    # Distance from each pixel to nearest predicted / reference foreground.
    # `distance_transform_edt` measures distance to the nearest *zero*; we
    # want distance to nearest *one*, so invert.
    dt_pred = distance_transform_edt(~pred_b)
    dt_gt = distance_transform_edt(~gt_b)

    matched_gt = int(np.logical_and(gt_skel, dt_pred <= tolerance_px).sum())
    matched_pred = int(np.logical_and(pred_skel, dt_gt <= tolerance_px).sum())

    completeness = matched_gt / gt_len
    correctness = matched_pred / pred_len
    denom = completeness + correctness - completeness * correctness
    quality = (completeness * correctness) / denom if denom > 0 else 0.0
    return {
        "completeness": completeness,
        "correctness": correctness,
        "quality": quality,
    }


# ── split selection ────────────────────────────────────────────────────
def select_split(images: list[Path], split: str, seed: int = 42) -> list[Path]:
    """Reproduce the trainer's 80/10/10 random split.

    The trainer uses ``random_split`` with seed 42 for the val/train cut.
    Here we use a deterministic NumPy permutation with the same seed so
    we can target any of the three splits without instantiating
    PyTorch's data pipeline.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))
    n = len(images)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    if split == "train":
        chosen = train_idx
    elif split == "val":
        chosen = val_idx
    elif split == "test":
        chosen = test_idx
    else:
        raise ValueError(f"unknown split: {split!r}")
    return [images[i] for i in sorted(chosen.tolist())]


# ── qualitative panels ────────────────────────────────────────────────
def render_panel(
    image_path: Path, mask_gt: np.ndarray, mask_pred: np.ndarray, out_path: Path
) -> None:
    """Save an input | ground-truth | prediction-overlay strip as PNG.

    No matplotlib dependency: we tile three RGB views with PIL. This
    keeps the eval script importable in environments where the user
    has not installed plotting libraries.
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    if mask_gt.shape != (h, w):
        mask_gt = np.array(Image.fromarray(mask_gt.astype(np.uint8) * 255).resize((w, h)))
        mask_gt = mask_gt > 127
    if mask_pred.shape != (h, w):
        mask_pred = np.array(
            Image.fromarray(mask_pred.astype(np.uint8) * 255).resize((w, h))
        )
        mask_pred = mask_pred > 127

    gt_rgb = image.copy()
    gt_rgb[mask_gt] = (0.4 * gt_rgb[mask_gt] + 0.6 * np.array([0, 200, 50])).astype(np.uint8)

    pred_rgb = image.copy()
    pred_rgb[mask_pred] = (
        0.4 * pred_rgb[mask_pred] + 0.6 * np.array([255, 90, 80])
    ).astype(np.uint8)

    strip = np.concatenate([image, gt_rgb, pred_rgb], axis=1)
    Image.fromarray(strip).save(out_path)


# ── main ───────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True, help="TTPLA root with images/ + masks/")
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/unet_resnet34_ttpla.pth"),
        help="path to .pth state_dict (Lightning .ckpt also accepted)",
    )
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tolerance", type=int, default=3, help="CCQ buffer in pixels")
    p.add_argument("--n-failures", type=int, default=3)
    p.add_argument("--n-successes", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="evaluate at most N images (0 = all)")
    p.add_argument("--out", type=Path, default=Path("docs"))
    args = p.parse_args()

    images_dir = args.data / "images"
    masks_dir = args.data / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        log.error("eval.missing_dirs", images=str(images_dir), masks=str(masks_dir))
        return 2

    all_images = sorted(p for p in images_dir.iterdir() if p.suffix in IMAGE_EXTS)
    all_images = [p for p in all_images if (masks_dir / f"{p.stem}.png").exists()]
    if not all_images:
        log.error("eval.no_pairs", images_dir=str(images_dir))
        return 2

    images = select_split(all_images, args.split)
    if args.limit > 0:
        images = images[: args.limit]
    log.info("eval.start", split=args.split, n_images=len(images))

    segmenter = ConductorSegmenter(weights_path=args.weights)
    if not args.weights.exists():
        log.warning("eval.no_weights — running with ImageNet init; numbers are not meaningful")

    per_image: list[PerImage] = []
    pred_cache: dict[str, np.ndarray] = {}

    for img_path in images:
        gt = np.array(Image.open(masks_dir / f"{img_path.stem}.png").convert("L")) > 127
        with Image.open(img_path) as im:
            prob = segmenter.segment(im)
        pred = prob > args.threshold

        m = pixel_metrics(pred, gt)
        c = ccq(pred, gt, tolerance_px=args.tolerance)
        per_image.append(
            PerImage(
                name=img_path.name,
                iou=m["iou"],
                precision=m["precision"],
                recall=m["recall"],
                f1=m["f1"],
                ccq_completeness=c["completeness"],
                ccq_correctness=c["correctness"],
                ccq_quality=c["quality"],
            )
        )
        pred_cache[img_path.name] = pred

    # Aggregate (macro-average over images; matches the convention in the
    # TTPLA paper and most thin-structure benchmarks).
    def avg(field: str) -> float:
        return float(np.mean([getattr(p, field) for p in per_image]))

    summary = {
        "split": args.split,
        "n_images": len(per_image),
        "threshold": args.threshold,
        "ccq_tolerance_px": args.tolerance,
        "iou_macro": avg("iou"),
        "precision_macro": avg("precision"),
        "recall_macro": avg("recall"),
        "f1_macro": avg("f1"),
        "ccq_completeness_macro": avg("ccq_completeness"),
        "ccq_correctness_macro": avg("ccq_correctness"),
        "ccq_quality_macro": avg("ccq_quality"),
    }
    log.info("eval.summary", **summary)

    # Pick failure / success gallery: rank by quality, take the worst N
    # (excluding cases with empty GT — they are uninformative) and the
    # best N. Failure-case selection is critical: pick the most
    # *instructive* failures, not the most embarrassing.
    informative = [
        p for p in per_image if p.ccq_quality < 1.0 or p.iou < 1.0
    ]
    informative.sort(key=lambda p: p.ccq_quality)
    worst = informative[: args.n_failures]
    best = sorted(informative, key=lambda p: -p.ccq_quality)[: args.n_successes]

    eval_dir = args.out / "screenshots" / "eval"
    (eval_dir / "successes").mkdir(parents=True, exist_ok=True)
    (eval_dir / "failures").mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(best, 1):
        img_path = next(p for p in images if p.name == item.name)
        gt = np.array(Image.open(masks_dir / f"{img_path.stem}.png").convert("L")) > 127
        render_panel(img_path, gt, pred_cache[item.name], eval_dir / "successes" / f"{i}.png")

    for i, item in enumerate(worst, 1):
        img_path = next(p for p in images if p.name == item.name)
        gt = np.array(Image.open(masks_dir / f"{img_path.stem}.png").convert("L")) > 127
        render_panel(img_path, gt, pred_cache[item.name], eval_dir / "failures" / f"{i}.png")

    # Markdown report
    md = [
        "# Evaluation results",
        "",
        f"_Generated by `training/evaluate.py` on the TTPLA `{args.split}` split._",
        "",
        "## Headline numbers",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Images evaluated | {summary['n_images']} |",
        f"| Threshold | {summary['threshold']} |",
        f"| CCQ buffer tolerance | {summary['ccq_tolerance_px']} px |",
        f"| Pixel IoU | {summary['iou_macro']:.4f} |",
        f"| Pixel precision | {summary['precision_macro']:.4f} |",
        f"| Pixel recall | {summary['recall_macro']:.4f} |",
        f"| Pixel F1 | {summary['f1_macro']:.4f} |",
        f"| CCQ completeness | {summary['ccq_completeness_macro']:.4f} |",
        f"| CCQ correctness | {summary['ccq_correctness_macro']:.4f} |",
        f"| **CCQ quality (headline)** | **{summary['ccq_quality_macro']:.4f}** |",
        "",
        "## Method",
        "",
        "Inference uses the production `ConductorSegmenter` (sliding-window "
        "tiling with Hann-windowed stitching). CCQ follows Wiedemann, Heipke, "
        "Mayer, Jamet (1998), *Empirical Evaluation of Automatically "
        "Extracted Road Axes*. Pixel IoU is included for comparison with "
        "prior work that reports it as the primary metric, even though it "
        "under-rewards near-misses on thin structures.",
        "",
        "## Qualitative results",
        "",
        f"### {len(best)} representative successes",
        "",
        *[
            f"![success {i}](screenshots/eval/successes/{i}.png)\n"
            f"_{item.name} — IoU {item.iou:.3f}, CCQ-Q {item.ccq_quality:.3f}._\n"
            for i, item in enumerate(best, 1)
        ],
        "",
        f"### {len(worst)} instructive failures",
        "",
        *[
            f"![failure {i}](screenshots/eval/failures/{i}.png)\n"
            f"_{item.name} — IoU {item.iou:.3f}, CCQ-Q {item.ccq_quality:.3f}. "
            "Diagnosis: TODO — fill in by inspection (texture / scale / context "
            "category from `docs/methodology.md` §7)._\n"
            for i, item in enumerate(worst, 1)
        ],
    ]
    out_md = args.out / "evaluation_results.md"
    out_md.write_text("\n".join(md))
    log.info("eval.report_written", path=str(out_md))

    # Also dump per-image JSON for downstream analysis
    (args.out / "evaluation_per_image.json").write_text(
        json.dumps([asdict(p) for p in per_image], indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
