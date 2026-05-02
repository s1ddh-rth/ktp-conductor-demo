"""Postprocessing: binary mask → skeleton → graph → simplified linestrings.

The pipeline:
  1. Threshold the probability map and clean small noise (morphological open).
  2. Skeletonize via medial-axis transform (Lee 1994 / scikit-image).
  3. Walk the skeleton, identifying junctions (3+ neighbours) and endpoints
     (1 neighbour) as graph nodes; pixel chains between them as edges.
  4. Simplify each edge with Douglas-Peucker (Shapely).

The graph is the right intermediate because it lets downstream modules
(hidden-cable inference, topology validation) reason about connectivity
rather than pixels.

Note on robustness
------------------
The custom skeleton walker is a sprint-grade implementation that handles
typical thin-cable masks well but can produce extra "junction" nodes on
unusually thick crossings. For production use, replace with `sknw` (a
mature library specifically for this task) or a more careful neighbour-
chain algorithm.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.ndimage import binary_opening
from shapely.geometry import LineString
from skimage.morphology import skeletonize


def mask_to_skeleton(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Probability map → cleaned 1-pixel-wide skeleton."""
    binary = (prob > threshold).astype(np.uint8)
    # Remove specks (radius-1 open)
    binary = binary_opening(binary, structure=np.ones((3, 3))).astype(np.uint8)
    return skeletonize(binary).astype(np.uint8)


def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """Walk the skeleton; build a graph with junction/endpoint nodes."""
    g = nx.Graph()
    h, w = skeleton.shape
    # Count 8-connected neighbours per skeleton pixel
    pad = np.pad(skeleton, 1, mode="constant")
    neigh = sum(
        np.roll(np.roll(pad, dy, 0), dx, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )[1:-1, 1:-1] * skeleton

    # Special points: endpoints (1 neighbour) and junctions (>2 neighbours)
    special = (skeleton > 0) & ((neigh == 1) | (neigh > 2))
    ys, xs = np.where(special)
    nodes = list(zip(xs.tolist(), ys.tolist(), strict=True))
    node_ids = {pt: i for i, pt in enumerate(nodes)}
    for pt in nodes:
        g.add_node(node_ids[pt], pos=pt)

    # Trace edges between special points
    visited = np.zeros_like(skeleton, dtype=bool)
    for sx, sy in nodes:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = sx + dx, sy + dy
                if not (0 <= nx_ < w and 0 <= ny_ < h):
                    continue
                if not skeleton[ny_, nx_] or visited[ny_, nx_]:
                    continue
                # Walk along the skeleton until we hit another special point
                path = [(sx, sy), (nx_, ny_)]
                visited[ny_, nx_] = True
                cx, cy = nx_, ny_
                while (cx, cy) not in node_ids:
                    found = False
                    for ey in (-1, 0, 1):
                        for ex in (-1, 0, 1):
                            if ex == 0 and ey == 0:
                                continue
                            tx, ty = cx + ex, cy + ey
                            if not (0 <= tx < w and 0 <= ty < h):
                                continue
                            if (
                                skeleton[ty, tx]
                                and not visited[ty, tx]
                                and (tx, ty) != (path[-2] if len(path) >= 2 else None)
                            ):
                                path.append((tx, ty))
                                visited[ty, tx] = True
                                cx, cy = tx, ty
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        break
                if (cx, cy) in node_ids and node_ids[(cx, cy)] != node_ids[(sx, sy)]:
                    g.add_edge(
                        node_ids[(sx, sy)],
                        node_ids[(cx, cy)],
                        path=path,
                        length=len(path),
                    )
    return g


def graph_to_linestrings(graph: nx.Graph, simplify_tol: float = 2.0) -> list[LineString]:
    """Convert each graph edge's pixel path into a simplified LineString."""
    lines: list[LineString] = []
    for _, _, data in graph.edges(data=True):
        coords = data.get("path", [])
        if len(coords) < 2:
            continue
        ls = LineString(coords).simplify(simplify_tol, preserve_topology=False)
        if ls.length > 5:  # drop tiny fragments
            lines.append(ls)
    return lines
