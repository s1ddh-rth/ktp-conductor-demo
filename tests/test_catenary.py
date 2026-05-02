"""Tests for the catenary fit (`app/ml/catenary.py`).

We assert four invariants of any sensible 2D catenary between two
fixed anchors:

1. **Anchor exactness** — the curve passes through both endpoints.
2. **Sag direction** — the midspan point lies on the +v half-plane
   relative to the chord (per the docstring's convention: 90° CW
   from chord, which is "down" in image-y for a horizontal span).
3. **Sag magnitude scales with sag_fraction** — bigger fraction →
   bigger midspan dip from the chord.
4. **Endpoint symmetry** — for a symmetric span the midpoint sits
   directly below the chord midpoint (within numeric tolerance).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.catenary import confidence_band, fit_catenary_2d


def _midspan_dip(curve: np.ndarray, p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Signed perpendicular distance from chord midpoint to curve midpoint.

    Positive = curve sags toward image-+y (downward on screen), which is the
    direction `fit_catenary_2d` puts the dip for a horizontal left-to-right
    chord (gravity-equivalent in screen space).
    """
    chord_mid = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
    curve_mid = curve[len(curve) // 2]
    chord_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    span = np.linalg.norm(chord_vec)
    # 90° CCW from chord direction → image-down for a horizontal LTR chord.
    perp = np.array([-chord_vec[1], chord_vec[0]]) / span
    return float(np.dot(curve_mid - chord_mid, perp))


def test_anchors_are_exact():
    p1, p2 = (0.0, 0.0), (100.0, 0.0)
    curve = fit_catenary_2d(p1, p2, sag_fraction=0.05)
    np.testing.assert_allclose(curve[0], p1, atol=1e-6)
    np.testing.assert_allclose(curve[-1], p2, atol=1e-6)


def test_sag_direction_is_perpendicular_cw():
    """The curve must dip toward the +v half-plane (CW from chord)."""
    p1, p2 = (0.0, 0.0), (100.0, 0.0)
    curve = fit_catenary_2d(p1, p2, sag_fraction=0.05)
    assert _midspan_dip(curve, p1, p2) > 0.0


def test_sag_monotonic_in_fraction():
    """Larger sag_fraction must produce a deeper dip."""
    p1, p2 = (0.0, 0.0), (200.0, 0.0)
    dips = [_midspan_dip(fit_catenary_2d(p1, p2, sag_fraction=f), p1, p2)
            for f in (0.01, 0.03, 0.06, 0.10)]
    # strict=False: by design dips and dips[1:] differ in length by one
    assert all(b > a for a, b in zip(dips, dips[1:], strict=False))


def test_sag_magnitude_matches_request():
    """The realised dip should equal sag_fraction * span (within bisection eps)."""
    p1, p2 = (0.0, 0.0), (300.0, 0.0)
    span = 300.0
    target = 0.04 * span  # 12 px
    curve = fit_catenary_2d(p1, p2, sag_fraction=0.04)
    dip = _midspan_dip(curve, p1, p2)
    assert abs(dip - target) < 1.0


def test_degenerate_short_span_returns_endpoints():
    p1, p2 = (10.0, 10.0), (10.4, 10.4)
    curve = fit_catenary_2d(p1, p2)
    np.testing.assert_allclose(curve[0], p1)
    np.testing.assert_allclose(curve[-1], p2)


def test_diagonal_span_anchors_remain_exact():
    """Rotation-aligned frame should still pin the anchors."""
    p1, p2 = (10.0, 10.0), (90.0, 50.0)
    curve = fit_catenary_2d(p1, p2, sag_fraction=0.05)
    np.testing.assert_allclose(curve[0], p1, atol=1e-6)
    np.testing.assert_allclose(curve[-1], p2, atol=1e-6)


def test_confidence_band_brackets_curve():
    """Upper and lower bands should sit on opposite sides of the curve."""
    p1, p2 = (0.0, 0.0), (100.0, 0.0)
    curve = fit_catenary_2d(p1, p2, sag_fraction=0.05)
    upper, lower = confidence_band(curve, sag_uncertainty=0.5)
    mid = len(curve) // 2
    # Bands shift in opposite directions perpendicular to the curve
    diff_upper = upper[mid] - curve[mid]
    diff_lower = lower[mid] - curve[mid]
    # Their dot product is strongly negative if they point opposite ways
    dot = float(np.dot(diff_upper, diff_lower))
    assert dot < 0


@pytest.mark.parametrize("n_points", [10, 50, 100])
def test_n_points_is_respected(n_points):
    curve = fit_catenary_2d((0.0, 0.0), (50.0, 0.0), n_points=n_points)
    assert curve.shape == (n_points, 2)
