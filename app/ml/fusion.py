"""RGB + LiDAR late fusion for conductor mapping.

Why this module exists
----------------------
The RGB segmenter and the LiDAR classifier in this prototype operate
independently — the segmenter ingests aerial imagery, the LiDAR
classifier ingests airborne point clouds. Each produces evidence of
"a conductor lives here", but neither uses the other's signal.

Real production conductor-mapping pipelines fuse the two modalities,
because each compensates for failure modes of the other:

- RGB confuses cables with similar thin-line features (fences,
  lane markings, roof seams) — but LiDAR-derived height-above-ground
  rules out anything close to the ground.
- LiDAR loses cables in vegetation occlusion — but RGB sees through
  light canopy when the cable's contrast is high.

This module implements the simplest credible fusion: **decision-level
late fusion** that re-scores RGB-derived LineStrings using LiDAR-
derived spatial features in a buffer around each line. It is
deliberately classical and interpretable — no learned cross-attention,
no joint embedding space — and serves as the *baseline* against
which a learned mid-fusion model would be evaluated.

Design lineage
--------------
Late fusion at the decision level is the canonical baseline in
multimodal sensing literature (Atrey et al. 2010, *Multimodal fusion
for multimedia analysis: a survey*). Mid-fusion via cross-attention
between modality-specific encoders (BEVFusion: Liu et al. 2023;
TransFusion: Bai et al. 2022) is a research-direction successor;
the methodology document names it as the Phase-2 work.

Limits made explicit
--------------------
1. Treats LiDAR as a noisy *binary attestation* per linestring, not
   as a continuous feature you backprop through. A learned model
   would do better.
2. Assumes co-registration between RGB and LiDAR coordinate frames.
   Real surveys ship camera-pose metadata that handles this; the
   demo accepts the LiDAR points already in image-pixel space.
3. The buffer width is a hyperparameter set heuristically. A
   production system would learn it (or use uncertainty-aware fusion
   per Kendall & Gal 2017).

References
----------
Atrey, P. K., Hossain, M. A., El Saddik, A., & Kankanhalli, M. S.
    (2010). Multimodal fusion for multimedia analysis: a survey.
    *Multimedia Systems*.
Liu, Z., et al. (2023). BEVFusion: Multi-Task Multi-Sensor Fusion
    with Unified Bird's-Eye View Representation. *ICRA*.
Bai, X., et al. (2022). TransFusion: Robust LiDAR-Camera Fusion for
    3D Object Detection with Transformers. *CVPR*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

from app.geo.lidar_features import compute_features


@dataclass
class FusedLineString:
    """A LineString re-scored with LiDAR-derived support evidence.

    Attributes
    ----------
    geometry : the original linestring (pixel-space coordinates).
    rgb_length_px : length of the linestring in pixels — proxy for
        the RGB segmenter's confidence in this run.
    lidar_linearity_support : mean linearity of LiDAR points within
        ``buffer_px`` of the line. High values (> 0.85) indicate the
        LiDAR independently agrees a 1D structure runs along this
        line. From Demantké et al. (2011)'s eigenvalue features.
    lidar_conductor_fraction : fraction of nearby LiDAR points that
        the eigenvalue classifier (``app.geo.lidar_features.classify``)
        labelled as "conductor" (class 2). High values (> 0.3) are
        strong corroboration.
    n_lidar_neighbours : number of LiDAR points found in the buffer.
        Low counts (< 5) make the support metrics unreliable.
    fused_confidence : aggregate score combining RGB length, LiDAR
        linearity, and LiDAR class agreement. In [0, 1].
    """

    geometry: LineString
    rgb_length_px: float
    lidar_linearity_support: float
    lidar_conductor_fraction: float
    n_lidar_neighbours: int
    fused_confidence: float


def _project_xy(points_xyz: np.ndarray) -> np.ndarray:
    """Drop the z-axis to get 2D positions in the image plane.

    Real surveys ship camera-pose metadata that maps 3D world
    coordinates into 2D image pixels. The prototype assumes the
    caller has already done that projection, so points arrive in
    image-pixel space (with z carrying height information).
    """
    return points_xyz[:, :2]


def fuse(
    linestrings: list[LineString],
    lidar_points: np.ndarray,
    lidar_classes: np.ndarray | None = None,
    buffer_px: float = 8.0,
    k_for_features: int = 12,
    min_neighbours: int = 5,
) -> list[FusedLineString]:
    """Re-score each RGB linestring using LiDAR evidence within a buffer.

    Parameters
    ----------
    linestrings
        Pixel-space LineStrings produced upstream by
        ``app.ml.postprocess.graph_to_linestrings``.
    lidar_points
        ``(N, 3)`` float array of LiDAR points already projected into
        image-pixel space. The third column holds height-above-ground
        used by ``compute_features``.
    lidar_classes
        Optional ``(N,)`` int array of per-point class labels from
        ``app.geo.lidar_features.classify``. Class 2 is "conductor".
        If absent, ``lidar_conductor_fraction`` is set to 0.
    buffer_px
        Spatial neighbourhood around each linestring within which
        LiDAR points contribute support. Default 8 px is reasonable
        for the prototype's synthetic geo-frame; in production this
        is a function of the survey's ground-sample distance.
    k_for_features
        k-nearest-neighbour parameter passed to ``compute_features``.
    min_neighbours
        Below this point count in the buffer, support metrics are
        considered unreliable and the fused confidence is biased
        toward the RGB-only signal.

    Returns
    -------
    list[FusedLineString], one per input linestring, in input order.

    Edge cases
    ----------
    Empty ``linestrings`` returns ``[]``. Empty ``lidar_points``
    returns FusedLineStrings with zero support metrics — the
    ``fused_confidence`` falls back to a length-normalised RGB-only
    score so the output is still meaningful.
    """
    if not linestrings:
        return []

    if lidar_points.size == 0:
        return [
            FusedLineString(
                geometry=ls,
                rgb_length_px=float(ls.length),
                lidar_linearity_support=0.0,
                lidar_conductor_fraction=0.0,
                n_lidar_neighbours=0,
                fused_confidence=_rgb_only_confidence(ls.length),
            )
            for ls in linestrings
        ]

    xy = _project_xy(lidar_points)
    features = compute_features(lidar_points, k=k_for_features)
    linearity = features["linearity"]

    out: list[FusedLineString] = []
    for ls in linestrings:
        # Spatial filter: which LiDAR points fall in the buffer?
        # Shapely's `buffer` + `within` is correct but quadratic on
        # point count; for the demo's ~150k subsampled cloud it's
        # acceptable. For production scale, switch to an STR-tree.
        buf = ls.buffer(buffer_px)
        in_buffer = np.array(
            [buf.contains(Point(x, y)) for x, y in xy], dtype=bool
        )
        n = int(in_buffer.sum())

        if n >= min_neighbours:
            mean_linearity = float(linearity[in_buffer].mean())
            cond_frac = (
                float((lidar_classes[in_buffer] == 2).mean())
                if lidar_classes is not None
                else 0.0
            )
        else:
            mean_linearity = 0.0
            cond_frac = 0.0

        out.append(
            FusedLineString(
                geometry=ls,
                rgb_length_px=float(ls.length),
                lidar_linearity_support=mean_linearity,
                lidar_conductor_fraction=cond_frac,
                n_lidar_neighbours=n,
                fused_confidence=_combine(ls.length, mean_linearity, cond_frac, n),
            )
        )
    return out


def _rgb_only_confidence(length_px: float) -> float:
    """Length-normalised RGB-only confidence (no LiDAR available).

    Squashes raw pixel length to ~[0, 1] using a logistic with an
    inflection at 60 px — typical visible cable run length in the
    sample TTPLA imagery. Empirically chosen, not learned.
    """
    return float(1.0 / (1.0 + np.exp(-(length_px - 60.0) / 30.0)))


def _combine(
    length_px: float,
    mean_linearity: float,
    cond_frac: float,
    n_neighbours: int,
) -> float:
    """Combine RGB length + LiDAR support into a single [0, 1] score.

    Weights (40 / 35 / 25) are heuristic, biased toward the RGB
    signal because the demo's LiDAR sample is a static curated tile,
    not co-registered with the user's uploaded image. They are
    deliberately documented here so a reviewer can see the choice
    is explicit, not learned.

    For production use, these weights would be fit on a held-out
    validation set with ground-truth fused-confidence labels (or
    derived from a calibrated learned scorer).
    """
    rgb = _rgb_only_confidence(length_px)
    lidar_geom = mean_linearity                       # in [0, 1]
    lidar_class = cond_frac                           # in [0, 1]

    # Reliability discount when neighbour count is low
    reliability = min(1.0, n_neighbours / 20.0)

    fused = (
        0.40 * rgb
        + 0.35 * lidar_geom * reliability
        + 0.25 * lidar_class * reliability
    )
    return float(np.clip(fused, 0.0, 1.0))
