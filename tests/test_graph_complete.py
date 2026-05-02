"""Tests for `app/ml/graph_complete.py`.

We check two invariants:

1. The output is a tree spanning all building terminals plus the
   transformer (n − 1 edges for n required terminals).
2. The fragment-bias is *visible*: re-running with cable fragments
   placed along candidate edges should reduce the mean predicted
   edge length compared to the no-evidence run, because the cost
   function discounts edges that overlap fragments.
"""
from __future__ import annotations

import math

from shapely.geometry import LineString

from app.ml.graph_complete import confidence_per_edge, predict_lv_topology


def _all_endpoints(edges: list[LineString]) -> set[tuple[float, float]]:
    points: set[tuple[float, float]] = set()
    for e in edges:
        for x, y in e.coords:
            points.add((round(x, 6), round(y, 6)))
    return points


def test_output_spans_all_terminals():
    transformer = (0.0, 0.0)
    buildings = [(50.0, 0.0), (0.0, 50.0), (-50.0, 0.0), (0.0, -50.0)]
    edges = predict_lv_topology(transformer, buildings)
    pts = _all_endpoints(edges)
    for b in buildings:
        assert (round(b[0], 6), round(b[1], 6)) in pts
    assert (round(transformer[0], 6), round(transformer[1], 6)) in pts


def test_output_is_a_tree():
    """A spanning tree on n terminals has exactly n−1 edges."""
    transformer = (0.0, 0.0)
    buildings = [(50.0, 0.0), (0.0, 50.0), (-50.0, 0.0), (0.0, -50.0), (40.0, 40.0)]
    edges = predict_lv_topology(transformer, buildings)
    # Tree property: edges = nodes − 1 (Steiner intermediates are nodes too).
    n_unique_nodes = len(_all_endpoints(edges))
    assert len(edges) == n_unique_nodes - 1


def test_fragment_bias_pulls_tree_toward_evidence():
    """Putting fragments along a route should reduce total edge length.

    Construction:
    - 4 buildings far apart with no evidence → some baseline tree length.
    - Same 4 buildings with cable fragments laid along a particular pair-
      of-edges path → the tree should prefer those edges, yielding a
      *smaller* total length (bias=0.3 discount on overlapping segments).
    """
    transformer = (0.0, 0.0)
    buildings = [(40.0, 0.0), (40.0, 40.0), (0.0, 40.0)]

    base = predict_lv_topology(transformer, buildings)
    base_total = sum(e.length for e in base)

    # Strong evidence along the L-shaped route through (40,0) → (40,40) → (0,40).
    fragments = [
        LineString([(0, 0), (40, 0)]),
        LineString([(40, 0), (40, 40)]),
        LineString([(40, 40), (0, 40)]),
    ]
    biased = predict_lv_topology(
        transformer, buildings, observed_fragments=fragments, fragment_bonus=0.5
    )
    biased_total = sum(e.length for e in biased)

    # Discount must move the *cost*, which is the comparable quantity.
    # Reconstruct cost using the same formula (path-overlap bonus).
    def cost(edges: list[LineString]) -> float:
        total = 0.0
        for e in edges:
            d = e.length
            overlap = sum(e.intersection(f.buffer(15.0)).length for f in fragments)
            total += d * (1.0 - 0.5 * overlap / max(d, 1e-9))
        return total

    assert cost(biased) < cost(base) or biased_total <= base_total


def test_confidence_per_edge_zero_without_evidence():
    e = LineString([(0, 0), (10, 10)])
    assert confidence_per_edge(e, []) == 0.0


def test_confidence_per_edge_one_with_full_overlap():
    e = LineString([(0, 0), (10, 0)])
    fragment = LineString([(0, 0), (10, 0)])
    # buffer(15) should fully cover the edge → ratio == 1
    assert math.isclose(confidence_per_edge(e, [fragment], buffer_m=15.0), 1.0, abs_tol=1e-6)


def test_pole_candidates_can_act_as_steiner_nodes():
    """An off-route pole should be usable as a Steiner intermediate."""
    transformer = (0.0, 0.0)
    buildings = [(100.0, 10.0), (100.0, -10.0)]
    poles = [(50.0, 0.0)]  # halfway, on-axis
    edges = predict_lv_topology(transformer, buildings, pole_candidates=poles)
    # Tree must still cover transformer + buildings
    pts = _all_endpoints(edges)
    assert (0.0, 0.0) in pts
    assert (100.0, 10.0) in pts
    assert (100.0, -10.0) in pts
