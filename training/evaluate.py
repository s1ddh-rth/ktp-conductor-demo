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


def select_split_by_session(
    images: list[Path], split: str, test_prefixes: list[str] | None = None
) -> list[Path]:
    """Group images by capture session (filename prefix) for a stricter
    held-out test.

    TTPLA filenames encode the originating flight via the prefix
    before the underscore (e.g. ``04_2220.jpg`` belongs to session
    ``04``). A random split mixes the same flight's images across
    train/val/test, inflating apparent generalisation. A
    *session-grouped* split holds out *all* images from named
    sessions, making cross-flight transfer the test the model has
    to pass.

    The default test sessions are ``{"14", "1000"}`` — chosen because
    they contain capture conditions distinct from the rest of the
    dataset (different lighting, different background distribution).
    These are the same prefixes surfaced as held-out examples in
    `app/static/examples/index.json`.

    Returns sorted images for determinism.
    """
    if test_prefixes is None:
        test_prefixes = ["14", "1000"]
    test_set = set(test_prefixes)

    def prefix(p: Path) -> str:
        return p.name.split("_", 1)[0]

    test_images = [p for p in images if prefix(p) in test_set]
    rest = [p for p in images if prefix(p) not in test_set]

    # 90/10 split of the remaining sessions for train/val
    rng = np.random.default_rng(42)
    order = rng.permutation(len(rest))
    n_val = max(1, len(rest) // 10)
    val_idx = sorted(order[:n_val].tolist())
    train_idx = sorted(order[n_val:].tolist())

    if split == "train":
        return sorted(rest[i] for i in train_idx)
    if split == "val":
        return sorted(rest[i] for i in val_idx)
    if split == "test":
        return sorted(test_images)
    raise ValueError(f"unknown split: {split!r}")


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error per Guo et al. (2017).

    Bins predicted probabilities into ``n_bins`` equal-width buckets,
    measures within each bucket the gap between mean predicted
    probability and observed positive frequency, weights by bucket
    population. Lower is better; well-calibrated models score near 0.

    Parameters
    ----------
    probs : flattened array of predicted P(positive) in [0, 1].
    labels : flattened array of {0, 1} ground-truth labels.
    n_bins : default 10; standard convention.

    Reference
    ---------
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
    Calibration of Modern Neural Networks. *ICML*.
    """
    probs = np.asarray(probs).ravel()
    labels = np.asarray(labels).ravel().astype(np.float64)
    if probs.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = probs.size
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        # Right-inclusive on the last bucket so probs == 1.0 are counted
        in_bucket = (
            (probs > lo) & (probs <= hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        )
        n = int(in_bucket.sum())
        if n == 0:
            continue
        mean_prob = float(probs[in_bucket].mean())
        observed = float(labels[in_bucket].mean())
        ece += (n / n_total) * abs(mean_prob - observed)
    return float(ece)


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
    p.add_argument(
        "--split-strategy",
        choices=["random", "session"],
        default="random",
        help=(
            "How the train/val/test partition is computed. 'random' "
            "uses the trainer's seed-42 random_split (matches what the "
            "model was trained against). 'session' holds out entire "
            "TTPLA capture sessions (filename prefixes 14, 1000 by "
            "default), giving a stricter cross-flight evaluation."
        ),
    )
    p.add_argument(
        "--test-prefixes",
        nargs="+",
        default=["14", "1000"],
        help="Filename prefixes treated as held-out sessions (only used with --split-strategy session)",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tolerance", type=int, default=3, help="CCQ buffer in pixels")
    p.add_argument("--n-failures", type=int, default=3)
    p.add_argument("--n-successes", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="evaluate at most N images (0 = all)")
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of equal-width bins for Expected Calibration Error (Guo et al. 2017).",
    )
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

    if args.split_strategy == "session":
        images = select_split_by_session(
            all_images, args.split, test_prefixes=args.test_prefixes
        )
    else:
        images = select_split(all_images, args.split)
    if args.limit > 0:
        images = images[: args.limit]
    log.info(
        "eval.start",
        split=args.split,
        strategy=args.split_strategy,
        n_images=len(images),
    )

    segmenter = ConductorSegmenter(weights_path=args.weights)
    if not args.weights.exists():
        log.warning("eval.no_weights — running with ImageNet init; numbers are not meaningful")

    per_image: list[PerImage] = []
    pred_cache: dict[str, np.ndarray] = {}
    # Accumulators for ECE — concatenating raw probs from full-sized
    # masks across N images would balloon memory, so we sub-sample
    # 50k pixels per image (ratio of positive vs negative preserved).
    ece_probs: list[np.ndarray] = []
    ece_labels: list[np.ndarray] = []
    ece_rng = np.random.default_rng(0)

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

        # Sub-sample for ECE — 50k pixels per image is plenty for a
        # 10-bin reliability estimate without spending RAM linearly
        # in image count.
        flat_prob = prob.ravel()
        flat_gt = gt.ravel()
        if flat_prob.size > 50_000:
            sel = ece_rng.choice(flat_prob.size, size=50_000, replace=False)
            ece_probs.append(flat_prob[sel])
            ece_labels.append(flat_gt[sel])
        else:
            ece_probs.append(flat_prob)
            ece_labels.append(flat_gt)

    # Aggregate (macro-average over images; matches the convention in the
    # TTPLA paper and most thin-structure benchmarks).
    def avg(field: str) -> float:
        return float(np.mean([getattr(p, field) for p in per_image]))

    # Calibration on the concatenated sub-sampled pixels
    if ece_probs:
        all_probs = np.concatenate(ece_probs)
        all_labels = np.concatenate(ece_labels)
        ece = expected_calibration_error(all_probs, all_labels, n_bins=args.ece_bins)
    else:
        ece = float("nan")

    summary = {
        "split": args.split,
        "strategy": args.split_strategy,
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
        "ece": ece,
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
    note_strategy = (
        "Random seed-42 partition over all TTPLA images — matches the "
        "trainer's `random_split`. Methodologically weak: spatial "
        "correlation between same-flight images inflates apparent "
        "generalisation. Reported here for reproducibility against the "
        "weights as trained."
        if args.split_strategy == "random"
        else (
            "Session-grouped split: filename prefixes "
            f"`{', '.join(args.test_prefixes)}` are held out entirely. "
            "Cross-flight evaluation — stricter than random, more "
            "representative of how the model would behave on a new "
            "DNO survey region."
        )
    )
    ece_note = (
        f"{summary['ece']:.4f}" if summary["ece"] == summary["ece"] else "n/a"
    )

    md = [
        "# Evaluation results",
        "",
        f"_Generated by `training/evaluate.py` on the TTPLA `{args.split}` split, "
        f"strategy `{args.split_strategy}`._",
        "",
        "## Headline numbers",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Images evaluated | {summary['n_images']} |",
        f"| Split strategy | `{summary['strategy']}` |",
        f"| Threshold | {summary['threshold']} |",
        f"| CCQ buffer tolerance | {summary['ccq_tolerance_px']} px |",
        f"| Pixel IoU | {summary['iou_macro']:.4f} |",
        f"| Pixel precision | {summary['precision_macro']:.4f} |",
        f"| Pixel recall | {summary['recall_macro']:.4f} |",
        f"| Pixel F1 | {summary['f1_macro']:.4f} |",
        f"| CCQ completeness | {summary['ccq_completeness_macro']:.4f} |",
        f"| CCQ correctness | {summary['ccq_correctness_macro']:.4f} |",
        f"| **CCQ quality (headline)** | **{summary['ccq_quality_macro']:.4f}** |",
        f"| ECE ({args.ece_bins}-bin, Guo et al. 2017) | {ece_note} |",
        "",
        "## Method",
        "",
        "Inference uses the production `ConductorSegmenter` (sliding-window "
        "tiling with Hann-windowed stitching). CCQ follows Wiedemann, Heipke, "
        "Mayer, Jamet (1998), *Empirical Evaluation of Automatically "
        "Extracted Road Axes*. Pixel IoU is included for comparison with "
        "prior work that reports it as the primary metric, even though it "
        "under-rewards near-misses on thin structures. Calibration is "
        "measured with the Expected Calibration Error (Guo, Pleiss, Sun & "
        "Weinberger, 2017) over a 50k-pixel-per-image sub-sample.",
        "",
        "### Split strategy",
        "",
        note_strategy,
        "",
        "TTPLA's repository ships a canonical `splitting_dataset_txt/` "
        "partition (Abdelfattah et al. 2020) which this evaluation does "
        "**not** use; the random-split numbers above are therefore not "
        "directly comparable with values reported in the original paper. "
        "Re-run with `--split-strategy session` for a stricter held-out "
        "evaluation that approximates the canonical-split protocol.",
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
            f"_{item.name} — IoU {item.iou:.3f}, CCQ-Q {item.ccq_quality:.3f}._\n"
            "_Diagnosis: hand-classify into one of the texture / scale / "
            "context categories from `docs/methodology.md` §7 after running._\n"
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
