"""Tests for `app/ml/fusion.py` — RGB + LiDAR late fusion.

Three invariants we check:

1. **No-LiDAR fallback**: when the LiDAR cloud is empty, the fused
   confidence falls back to a length-normalised RGB-only score and
   the support metrics are zero. Output length matches input.
2. **Cable-aligned LiDAR boosts confidence**: a synthetic cloud
   containing a high-linearity line co-located with an RGB
   linestring should produce higher fused confidence than the same
   linestring with no LiDAR support.
3. **Far-away LiDAR is ignored**: LiDAR points outside the buffer
   (or with low neighbour counts) shouldn't influence the score.

Plus a smoke test that the FastAPI router accepts a request and
returns a sensible response shape — exercising the full
RGB→postprocess→fuse path on a tiny synthetic image.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from shapely.geometry import LineString

from app.main import app
from app.ml.fusion import fuse


def _cable_along_x(n: int = 400, y: float = 50.0, length: float = 100.0) -> np.ndarray:
    """Synthetic LiDAR cable: linear in x, constant y, elevated z."""
    x = np.linspace(0, length, n)
    pts = np.stack([x, np.full(n, y), np.full(n, 8.0)], axis=1)
    pts += 0.05 * np.random.default_rng(0).standard_normal(pts.shape)
    return pts.astype(np.float32)


def _ground_plane(n: int = 1000, side: float = 200.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, side, size=(n, 2))
    z = 0.05 * rng.standard_normal(n)
    return np.column_stack([xy, z]).astype(np.float32)


def test_empty_lidar_returns_rgb_only():
    ls = LineString([(0, 50), (100, 50)])
    out = fuse([ls], lidar_points=np.empty((0, 3), dtype=np.float32))
    assert len(out) == 1
    f = out[0]
    assert f.lidar_linearity_support == 0.0
    assert f.lidar_conductor_fraction == 0.0
    assert f.n_lidar_neighbours == 0
    assert 0.0 <= f.fused_confidence <= 1.0
    # 100-pixel line should be in the upper half of the logistic
    assert f.fused_confidence > 0.4


def test_aligned_lidar_boosts_fused_confidence():
    """A linestring aligned with a synthetic LiDAR cable should score
    higher than one with no LiDAR support."""
    ls = LineString([(0, 50), (100, 50)])
    cable = _cable_along_x()
    classes = np.full(len(cable), 2, dtype=np.int8)  # all "conductor"

    no_support = fuse([ls], lidar_points=np.empty((0, 3), dtype=np.float32))[0]
    with_support = fuse([ls], lidar_points=cable, lidar_classes=classes, buffer_px=10.0)[0]
    assert with_support.fused_confidence > no_support.fused_confidence
    assert with_support.lidar_linearity_support > 0.85
    assert with_support.lidar_conductor_fraction > 0.8


def test_distant_lidar_is_ignored():
    """LiDAR points far from the linestring shouldn't contribute."""
    ls = LineString([(0, 50), (100, 50)])
    far_cable = _cable_along_x()  # at y=50
    far_cable[:, 1] = 200.0       # move it to y=200, well outside buffer
    classes = np.full(len(far_cable), 2, dtype=np.int8)
    out = fuse([ls], lidar_points=far_cable, lidar_classes=classes, buffer_px=10.0)[0]
    assert out.n_lidar_neighbours == 0
    assert out.lidar_linearity_support == 0.0


def test_empty_linestrings_returns_empty():
    assert fuse([], lidar_points=_ground_plane()) == []


def test_output_preserves_input_order():
    a = LineString([(0, 10), (50, 10)])
    b = LineString([(0, 90), (50, 90)])
    c = LineString([(0, 50), (50, 50)])
    cable = _cable_along_x()
    cable[:, 1] = 50.0  # support `c`, not the others
    out = fuse([a, b, c], lidar_points=cable, buffer_px=8.0)
    assert out[0].geometry.equals(a)
    assert out[1].geometry.equals(b)
    assert out[2].geometry.equals(c)
    # Only `c` should pick up support
    assert out[2].n_lidar_neighbours > out[0].n_lidar_neighbours
    assert out[2].n_lidar_neighbours > out[1].n_lidar_neighbours


# ── router smoke test ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_png_bytes(size: tuple[int, int] = (96, 96)) -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_fuse_endpoint_returns_geojson(client):
    """End-to-end: POST an image, expect GeoJSON with fused properties.

    No LiDAR sample is on disk in CI — the endpoint should still
    return a valid response, with `lidar_available: false` flagged.
    """
    png = _make_png_bytes((96, 96))
    r = client.post("/api/fuse", files={"file": ("t.png", png, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 96
    assert body["height"] == 96
    assert body["geojson"]["type"] == "FeatureCollection"
    assert "summary" in body
    assert "lidar_available" in body["summary"]
    assert "n_linestrings" in body["summary"]
