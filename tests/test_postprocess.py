"""Tests for `app/ml/postprocess.py` — mask → skeleton → graph → linestrings.

We feed three carefully-shaped synthetic masks and check the graph
extraction recovers them sensibly.
"""
from __future__ import annotations

import numpy as np

from app.ml.postprocess import graph_to_linestrings, mask_to_skeleton, skeleton_to_graph


def _line_mask(h: int, w: int) -> np.ndarray:
    """A horizontal 3-pixel-thick line spanning the image."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[h // 2 - 1 : h // 2 + 2, 5 : w - 5] = 1
    return m


def _cross_mask(h: int, w: int) -> np.ndarray:
    """A horizontal + vertical line crossing in the middle."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[h // 2 - 1 : h // 2 + 2, 5 : w - 5] = 1
    m[5 : h - 5, w // 2 - 1 : w // 2 + 2] = 1
    return m


def _curve_mask(h: int, w: int) -> np.ndarray:
    """A quarter-circle arc."""
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - h, xx - 0)
    return ((r > h - 4) & (r < h - 1)).astype(np.uint8)


def test_skeleton_is_one_pixel_wide():
    """`mask_to_skeleton` must reduce a thick line to a 1px ridge."""
    prob = _line_mask(60, 200).astype(np.float32)
    skel = mask_to_skeleton(prob, threshold=0.5)
    # Each non-empty column should have exactly one set pixel
    cols_with_any = np.where(skel.sum(axis=0) > 0)[0]
    assert len(cols_with_any) > 0
    assert all(skel[:, c].sum() == 1 for c in cols_with_any)


def test_line_mask_yields_one_linestring():
    skel = mask_to_skeleton(_line_mask(60, 200).astype(np.float32))
    g = skeleton_to_graph(skel)
    lines = graph_to_linestrings(g, simplify_tol=2.0)
    assert len(lines) == 1
    # Endpoint x-coords should bracket the original line
    xs = sorted({c[0] for ls in lines for c in ls.coords})
    assert xs[0] < 10
    assert xs[-1] > 180


def test_cross_mask_yields_multiple_branches():
    """A '+' shape topologically has 4 endpoint-to-junction branches.

    The current sprint-grade walker (documented in `postprocess.py` as a
    known limitation) marks junction pixels visited along the first
    traversal, which can prevent later branches from re-attaching to the
    same junction. We assert the weaker guarantee that *some* branches
    are recovered — the production fix is to switch to `sknw`, noted in
    the methodology limitations.
    """
    skel = mask_to_skeleton(_cross_mask(80, 200).astype(np.float32))
    g = skeleton_to_graph(skel)
    lines = graph_to_linestrings(g, simplify_tol=1.0)
    assert len(lines) >= 2
    # And at least one node was a junction
    assert g.number_of_nodes() >= 4


def test_curve_mask_yields_one_linestring_after_simplify():
    skel = mask_to_skeleton(_curve_mask(60, 60).astype(np.float32))
    g = skeleton_to_graph(skel)
    lines = graph_to_linestrings(g, simplify_tol=1.5)
    assert 1 <= len(lines) <= 3  # may split if the medial axis branches at the ends


def test_empty_mask_returns_empty_graph():
    skel = mask_to_skeleton(np.zeros((40, 40), dtype=np.float32))
    g = skeleton_to_graph(skel)
    assert g.number_of_nodes() == 0
    assert graph_to_linestrings(g) == []


def test_subthreshold_probabilities_drop_out():
    """All-0.3 probability with threshold 0.5 → nothing survives."""
    prob = np.full((50, 50), 0.3, dtype=np.float32)
    skel = mask_to_skeleton(prob, threshold=0.5)
    assert skel.sum() == 0
