"""Tests for `app/geo/lidar_features.py`.

For each of three hand-constructed point clouds — a cable line, a
horizontal ground plane, a vertical pole — we check that:

- The eigenvalue-derived feature matches the expected geometry
  (high linearity for cables, high planarity for ground, high
  verticality for poles).
- The integer class returned by `classify` is the expected one for
  the *majority* of points.
"""
from __future__ import annotations

import numpy as np

from app.geo.lidar_features import classify, compute_features

RNG = np.random.default_rng(0)


def _cable_line(n: int = 800, length: float = 30.0, height: float = 8.0) -> np.ndarray:
    t = np.linspace(0, length, n)
    pts = np.stack(
        [t, np.full(n, 0.0), np.full(n, height)], axis=1
    ) + 0.02 * RNG.standard_normal((n, 3))
    return pts.astype(np.float32)


def _ground_plane(n: int = 4000, side: float = 30.0) -> np.ndarray:
    xy = RNG.uniform(-side / 2, side / 2, size=(n, 2))
    z = 0.05 * RNG.standard_normal(n)
    return np.column_stack([xy, z]).astype(np.float32)


def _vertical_pole(n: int = 400, height: float = 12.0, noise: float = 0.01) -> np.ndarray:
    z = np.linspace(0, height, n)
    pts = np.stack(
        [np.full(n, 0.0), np.full(n, 0.0), z], axis=1
    ) + noise * RNG.standard_normal((n, 3))
    return pts.astype(np.float32)


def test_cable_line_has_high_linearity():
    pts = _cable_line()
    feat = compute_features(pts, k=8)
    assert feat["linearity"].mean() > 0.95
    assert feat["planarity"].mean() < 0.05


def test_ground_plane_has_high_planarity():
    pts = _ground_plane()
    feat = compute_features(pts, k=20)
    # Larger k smooths the eigenvalue estimate. Threshold accounts for the
    # 0.05 noise-to-extent ratio and edge points where neighbour balls
    # are not isotropic.
    assert feat["planarity"].mean() > 0.5


def test_pole_has_high_verticality():
    pts = _vertical_pole(n=600)
    feat = compute_features(pts, k=12)
    # Pole points are linear (1D structure) and vertical (small |n_z|).
    assert feat["linearity"].mean() > 0.85
    assert feat["verticality"].mean() > 0.7


def test_classify_assigns_conductor_to_cable_with_ground_reference():
    """Conductor classification needs a ground reference for height-above-ground.

    `classify` estimates ground as the 5th percentile of z. A cable-only
    cloud at uniform z gives `height = 0` everywhere and never passes the
    `height > 3` threshold, so we add a thin ground plane below the cable
    to anchor the percentile.
    """
    cable = _cable_line()
    ground = _ground_plane(n=2000)
    cloud = np.concatenate([cable, ground], axis=0)
    classes = classify(cloud, k=12)
    cable_classes = classes[: len(cable)]
    assert int(np.bincount(cable_classes, minlength=4).argmax()) == 2


def test_classify_assigns_ground_to_flat_majority():
    ground = _ground_plane()
    classes = classify(ground, k=12)
    # ground class (0) — vegetation requires height > 1; a flat plane stays at 0
    assert int(np.bincount(classes, minlength=4).argmax()) == 0


def test_classify_handles_combined_cloud():
    """A mixed cloud should label each component with the right majority."""
    cable = _cable_line()
    ground = _ground_plane(n=2000)
    pole = _vertical_pole()
    combined = np.concatenate([cable, ground, pole], axis=0)
    classes = classify(combined, k=12)
    # Cable points are at indices 0:len(cable)
    cable_classes = classes[: len(cable)]
    ground_classes = classes[len(cable) : len(cable) + len(ground)]
    assert int(np.bincount(cable_classes, minlength=4).argmax()) == 2
    assert int(np.bincount(ground_classes, minlength=4).argmax()) == 0
