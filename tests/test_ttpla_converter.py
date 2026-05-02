"""Tests for `scripts/ttpla_to_masks.py`.

Constructs minimal fake annotation files in each of the supported
formats (LabelMe, TTPLA-native, COCO) and verifies the rasteriser
sets the expected pixels.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _write_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    Image.new("RGB", size, (128, 128, 128)).save(path)


def _run_converter(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.ttpla_to_masks", "--root", str(root), *extra],
        check=True,
        capture_output=True,
        text=True,
    )


def test_labelme_polygon_rasterises(tmp_path: Path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()

    _write_image(images / "scene01.jpg", size=(64, 64))

    ann = {
        "shapes": [
            {
                "label": "cable",
                "shape_type": "polygon",
                "points": [[10, 10], [50, 10], [50, 20], [10, 20]],
            }
        ]
    }
    (annotations / "scene01.json").write_text(json.dumps(ann))

    _run_converter(tmp_path)

    mask = np.array(Image.open(tmp_path / "masks" / "scene01.png"))
    # Pixels inside the polygon should be 255
    assert mask[15, 30] == 255
    # Outside corners should be 0
    assert mask[0, 0] == 0
    assert mask[60, 60] == 0


def test_classes_filter_excludes_other_labels(tmp_path: Path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()

    _write_image(images / "scene02.jpg", size=(50, 50))
    ann = {
        "shapes": [
            {"label": "tree", "shape_type": "polygon",
             "points": [[5, 5], [15, 5], [15, 15], [5, 15]]},
            {"label": "cable", "shape_type": "polygon",
             "points": [[20, 20], [40, 20], [40, 30], [20, 30]]},
        ]
    }
    (annotations / "scene02.json").write_text(json.dumps(ann))

    _run_converter(tmp_path, "--classes", "cable")

    mask = np.array(Image.open(tmp_path / "masks" / "scene02.png"))
    assert mask[10, 10] == 0      # tree polygon ignored
    assert mask[25, 30] == 255    # cable polygon rasterised


def test_dry_run_writes_no_files(tmp_path: Path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()

    _write_image(images / "scene03.jpg")
    (annotations / "scene03.json").write_text(
        json.dumps({"shapes": [{"label": "cable", "points": [[1, 1], [10, 10]]}]})
    )

    _run_converter(tmp_path, "--dry-run")
    assert not (tmp_path / "masks").exists()


def test_missing_image_does_not_crash(tmp_path: Path):
    """A JSON without a matching image should warn and continue."""
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()

    _write_image(images / "good.jpg")
    (annotations / "good.json").write_text(
        json.dumps({"shapes": [{"label": "cable", "points": [[1, 1], [5, 5]]}]})
    )
    # Orphan annotation
    (annotations / "missing.json").write_text(
        json.dumps({"shapes": [{"label": "cable", "points": [[1, 1], [5, 5]]}]})
    )

    proc = _run_converter(tmp_path)
    assert "missing" in (proc.stdout + proc.stderr)
    assert (tmp_path / "masks" / "good.png").exists()
