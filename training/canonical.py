"""TTPLA canonical split loader.

Shared by ``training/train.py`` and ``training/evaluate.py`` so both
consume the same train/val/test partition. The dataset's authors ship
the canonical three-way split as

    splitting_dataset_txt/train.txt    # 905 entries
    splitting_dataset_txt/val.txt      # 109 entries
    splitting_dataset_txt/test.txt     # 220 entries

totalling 1234 images. The released TTPLA directory has grown to 1242
images; the 8 unassigned post-publication additions are excluded to
preserve split integrity.

Each line of the splits files is a JSON annotation filename
(e.g. ``53_00231.json``). We strip the ``.json`` suffix to recover
the bare basename, then resolve against the image directory.

Reference: Abdelfattah, R., Wang, X., & Wang, S. (2020). TTPLA: An
Aerial-Image Dataset for Detection and Segmentation of Transmission
Towers and Power Lines. *ACCV*.
"""
from __future__ import annotations

from pathlib import Path

# Image extensions the loader will pick up. TTPLA releases vary in
# whether they ship .jpg or .JPG; we match all common variants.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Canonical split sizes from upstream's splitting_dataset_txt/.
CANONICAL_TRAIN = 905
CANONICAL_VAL = 109
CANONICAL_TEST = 220
CANONICAL_TOTAL = CANONICAL_TRAIN + CANONICAL_VAL + CANONICAL_TEST  # 1234


def discover_images(images_dir: Path) -> list[Path]:
    """List image files in ``images_dir`` (non-recursive, sorted)."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"no images in {images_dir}")
    return paths


def _read_splits_file(path: Path) -> list[str]:
    """Read a TTPLA splits file, stripping whitespace and the .json suffix.

    Each line of train.txt / val.txt / test.txt is a JSON annotation
    filename (e.g. ``53_00231.json``); we keep the bare basename for
    matching against the corresponding ``.jpg``.
    """
    if not path.exists():
        raise FileNotFoundError(f"canonical split file missing: {path}")
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith(".json"):
            s = s[: -len(".json")]
        out.append(s)
    return out


def load_canonical_splits(
    splits_dir: Path,
    images_dir: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Load TTPLA's official canonical train / val / test partition.

    Reads ``train.txt`` (905), ``val.txt`` (109), and ``test.txt`` (220)
    from ``splits_dir``, resolves each basename to a ``.jpg`` in
    ``images_dir``, and returns ``(train_paths, val_paths, test_paths)``.

    Verification checks are printed to stdout and a ``RuntimeError`` is
    raised if any of them fails. The intent is to fail loudly during
    training startup if the dataset on disk has drifted from the
    canonical partition (e.g. a partial download or a renamed file)
    rather than silently train on an unintended subset.
    """
    train_list = _read_splits_file(splits_dir / "train.txt")
    val_list = _read_splits_file(splits_dir / "val.txt")
    test_list = _read_splits_file(splits_dir / "test.txt")

    train_set = set(train_list)
    val_set = set(val_list)
    test_set = set(test_list)
    actual_total = len(train_list) + len(val_list) + len(test_list)

    image_lookup = {p.stem: p for p in discover_images(images_dir)}

    def _resolve(names: list[str]) -> tuple[list[Path], list[str]]:
        found: list[Path] = []
        missing: list[str] = []
        for n in names:
            p = image_lookup.get(n)
            if p is None:
                missing.append(n)
            else:
                found.append(p)
        return found, missing

    train_paths, train_missing = _resolve(train_list)
    val_paths, val_missing = _resolve(val_list)
    test_paths, test_missing = _resolve(test_list)
    all_missing = train_missing + val_missing + test_missing

    print(
        f"[canonical splits] train: {len(train_list)} | "
        f"val: {len(val_list)} | test: {len(test_list)}"
    )
    print("[canonical splits] verification:")
    checks: list[tuple[str, bool, str]] = [
        (
            f"len(train) == {CANONICAL_TRAIN}",
            len(train_list) == CANONICAL_TRAIN,
            f"actual {len(train_list)}",
        ),
        (
            f"len(val) == {CANONICAL_VAL}",
            len(val_list) == CANONICAL_VAL,
            f"actual {len(val_list)}",
        ),
        (
            f"len(test) == {CANONICAL_TEST}",
            len(test_list) == CANONICAL_TEST,
            f"actual {len(test_list)}",
        ),
        (
            f"len(train)+len(val)+len(test) == {CANONICAL_TOTAL}",
            actual_total == CANONICAL_TOTAL,
            f"actual {actual_total}",
        ),
        ("train ∩ val == ∅", not (train_set & val_set), f"{len(train_set & val_set)} overlap"),
        ("train ∩ test == ∅", not (train_set & test_set), f"{len(train_set & test_set)} overlap"),
        ("val ∩ test == ∅", not (val_set & test_set), f"{len(val_set & test_set)} overlap"),
        (
            "every basename resolves to a {basename}.jpg under images_dir",
            not all_missing,
            f"{len(all_missing)} missing; first 5: {all_missing[:5]}",
        ),
    ]
    failures: list[str] = []
    for label, ok, detail in checks:
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {label}  ({detail})")
        if not ok:
            failures.append(f"{label} ({detail})")

    if failures:
        raise RuntimeError(
            "canonical-splits verification failed:\n  - " + "\n  - ".join(failures)
        )

    return train_paths, val_paths, test_paths
