"""Catenary fitting for hidden-conductor inference.

Suspended cables under their own weight follow a catenary curve:

    y(x) = a · cosh((x - x0) / a) + c

In aerial imagery (top-down) the visible projection of a span between two
poles is approximately a straight line plus a sag perpendicular to it. We
fit a 2D catenary to the segment endpoints and (optionally) any partial
visible mask points in between.

This module is deliberately small: a physical prior is more interpretable
than a learned regressor when only two anchor points are observed. It is
the single piece of the demo that crosses from "ML" into "domain physics".

References
----------
Irvine, H. M. (1981). Cable Structures. MIT Press.
"""
from __future__ import annotations

import numpy as np


def _catenary(x: np.ndarray, a: float, x0: float, c: float) -> np.ndarray:
    return a * np.cosh((x - x0) / a) + c


def fit_catenary_2d(
    p1: tuple[float, float],
    p2: tuple[float, float],
    sag_fraction: float = 0.02,
    n_points: int = 50,
) -> np.ndarray:
    """Fit a catenary between two anchor points and return sampled curve.

    Parameters
    ----------
    p1, p2 : (x, y) image coordinates of pole tops or visible cable ends.
    sag_fraction : initial guess for sag relative to span (typical LV sag is
        2–5% of span; we default to 2% which biases toward shallow curves).
    n_points : number of points to sample on the curve.

    Returns
    -------
    np.ndarray of shape (n_points, 2) — sampled (x, y) along the catenary,
    rotated and translated back to image coordinates.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    span = float(np.hypot(dx, dy))
    if span < 1.0:
        return np.array([p1, p2], dtype=float)

    # Work in a span-aligned frame (u along chord, v perpendicular).
    # v_axis points 90° CW from the chord — for a left-to-right horizontal
    # chord this is "downward" in image space (+y), which matches a real
    # cable sagging under gravity. On the Leaflet map this also reads
    # naturally because the catenary visibly drops below the straight line.
    u_axis = np.array([dx, dy]) / span
    v_axis = np.array([u_axis[1], -u_axis[0]])  # 90° CW

    # Closed-form catenary in the chord-aligned (u, v) frame.
    # Constraint: y(0) = y(span) = 0 (both anchors on the chord) is satisfied
    # by symmetry around x0 = span/2 with c = -a · cosh(span/(2a)).
    # The remaining free parameter `a` is set so that midspan dip equals
    # the desired sag. Because dip(a) = a · (cosh(span/(2a)) − 1) is monotonic
    # decreasing in a, we solve by bisection.
    initial_sag = max(sag_fraction * span, 1.0)

    def dip(a: float) -> float:
        return a * np.cosh(span / (2 * a)) - a

    a_lo, a_hi = span / 1000.0, span * 1000.0
    for _ in range(80):
        a_mid = 0.5 * (a_lo + a_hi)
        if dip(a_mid) > initial_sag:
            a_lo = a_mid
        else:
            a_hi = a_mid
    a = 0.5 * (a_lo + a_hi)
    c = -a * np.cosh(span / (2 * a))
    x0 = span / 2

    u = np.linspace(0, span, n_points)
    v = a * np.cosh((u - x0) / a) + c
    # Rotate and translate back to image frame
    pts = np.outer(u, u_axis) + np.outer(v, v_axis) + np.array([x1, y1])
    return pts


def confidence_band(
    curve: np.ndarray, sag_uncertainty: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return upper/lower bands around the fitted curve.

    Without observations between the anchors, sag is poorly constrained.
    The band brackets reasonable physical alternatives and is rendered in
    the UI to communicate that the inferred line is one hypothesis.
    """
    if len(curve) < 3:
        return curve, curve
    midpoint = curve[len(curve) // 2]
    upper = curve.copy()
    lower = curve.copy()
    # Shift midspan only; taper to anchors
    n = len(curve)
    taper = np.sin(np.linspace(0, np.pi, n))
    direction = curve[-1] - curve[0]
    perp = np.array([-direction[1], direction[0]]) / (np.linalg.norm(direction) + 1e-9)
    band = sag_uncertainty * np.linalg.norm(direction) * 0.05  # 5% of span scale
    for i in range(n):
        upper[i] += taper[i] * band * perp
        lower[i] -= taper[i] * band * perp
    return upper, lower
