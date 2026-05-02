"""Topology completion: infer missing LV cable runs from partial evidence.

Problem framing
---------------
Given:
  - A set of building footprints (each must connect to the LV network).
  - A set of known infrastructure points (transformers, poles, pillars).
  - A set of partially observed cable fragments from the segmenter.

Predict the connectivity of the LV network.

Method (deliberately interpretable for the demo)
------------------------------------------------
1. Build a candidate graph: nodes = buildings ∪ poles ∪ transformers.
2. Edge weights penalise long jumps and reward proximity to observed
   fragments (so the predicted topology hugs visible evidence).
3. Compute a Steiner-tree approximation using NetworkX, rooted at the
   transformer. Approximation: minimum spanning tree on the metric closure
   of mandatory nodes (Kou-Markowsky-Berman).

For real LV inference, a graph neural network conditioned on these inputs
is the obvious next step (GraphSAGE / GAT). The MST baseline is fast,
deterministic, and good enough to communicate the *idea* of the method.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point


def predict_lv_topology(
    transformer: tuple[float, float],
    buildings: list[tuple[float, float]],
    observed_fragments: list[LineString] | None = None,
    pole_candidates: list[tuple[float, float]] | None = None,
    fragment_bonus: float = 0.3,
) -> list[LineString]:
    """Predict LV cable network as a tree rooted at the transformer.

    Returns LineStrings representing predicted edges.
    """
    observed_fragments = observed_fragments or []
    pole_candidates = pole_candidates or []
    nodes: list[tuple[float, float]] = [transformer] + buildings + pole_candidates
    g = nx.Graph()
    for i, n in enumerate(nodes):
        g.add_node(i, pos=n)

    # Pre-buffer observed fragments for fast nearness check
    fragment_buffers = [f.buffer(15.0) for f in observed_fragments]

    def edge_cost(a: tuple[float, float], b: tuple[float, float]) -> float:
        midline = LineString([a, b])
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        # Reward overlap with observed fragments
        overlap_len = 0.0
        for buf in fragment_buffers:
            inter = midline.intersection(buf)
            if not inter.is_empty:
                overlap_len += inter.length
        if midline.length > 0:
            ratio = overlap_len / midline.length
            d *= 1.0 - fragment_bonus * ratio
        return d

    # Dense candidate edges (k-nearest, to bound complexity)
    k = min(8, len(nodes) - 1)
    coords = np.array(nodes)
    for i in range(len(nodes)):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        nearest = np.argsort(dists)[1 : k + 1]
        for j in nearest:
            if not g.has_edge(i, int(j)):
                g.add_edge(i, int(j), weight=edge_cost(nodes[i], nodes[int(j)]))

    # Mandatory terminals: transformer (0) and all buildings
    terminals = list(range(1 + len(buildings)))
    try:
        tree = nx.algorithms.approximation.steiner_tree(g, terminals, weight="weight")
    except Exception:  # fall back to MST on the candidate graph
        tree = nx.minimum_spanning_tree(g, weight="weight")

    return [
        LineString([g.nodes[u]["pos"], g.nodes[v]["pos"]]) for u, v in tree.edges()
    ]


def confidence_per_edge(
    edge: LineString, observed_fragments: list[LineString], buffer_m: float = 15.0
) -> float:
    """Fraction of an edge supported by observed segmentation evidence."""
    if edge.length < 1e-6:
        return 0.0
    supported = 0.0
    for f in observed_fragments:
        inter = edge.intersection(f.buffer(buffer_m))
        if not inter.is_empty:
            supported += inter.length
    return min(1.0, supported / edge.length)
