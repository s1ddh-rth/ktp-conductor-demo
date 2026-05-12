"""POST /api/fuse — RGB segmentation + LiDAR re-scoring → fused GeoJSON.

Pipeline:
  1. Segment the uploaded RGB image with the production segmenter.
  2. Postprocess to LineStrings via the existing skeleton walker.
  3. Load the curated LiDAR sample (same tile served by /api/lidar).
  4. Re-score each LineString with LiDAR-derived features.
  5. Emit GeoJSON with fused-confidence + per-modality support
     properties on each feature, plus a summary block.

This endpoint exists primarily to demonstrate that the prototype
implements decision-level RGB+LiDAR fusion — the JD Essential that
each modality alone does not satisfy. See ``app/ml/fusion.py`` for
the design rationale and the references back to BEVFusion /
TransFusion as Phase-2 mid-fusion follow-ons.
"""
from __future__ import annotations

import io

import laspy
import numpy as np
import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image

from app.config import settings
from app.geo.lidar_features import classify
from app.geo.vectorise import linestrings_to_geojson
from app.limiter import limiter
from app.ml.fusion import fuse
from app.ml.postprocess import graph_to_linestrings, mask_to_skeleton, skeleton_to_graph

router = APIRouter()
log = structlog.get_logger()

MAX_BYTES = settings.max_upload_mb * 1024 * 1024


def _load_lidar_sample(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and classify the curated LiDAR tile.

    Mirrors the loader in ``app.routers.lidar`` so both endpoints
    return the same scene; the classifier output is reused.
    """
    path = settings.examples_dir / f"{name}.laz"
    if not path.exists():
        path = settings.examples_dir / f"{name}.las"
    if not path.exists():
        raise FileNotFoundError(f"LiDAR sample '{name}' not on disk")

    with laspy.open(path) as fh:
        las = fh.read()
    pts = np.column_stack([las.x, las.y, las.z]).astype(np.float32)

    # Re-centre + downsample for tractability — same convention as
    # the LiDAR endpoint so the two views are consistent.
    if len(pts) > 80_000:
        idx = np.random.default_rng(0).choice(len(pts), size=80_000, replace=False)
        pts = pts[idx]
    pts -= pts.mean(axis=0)
    classes = classify(pts, k=12)
    return pts, classes


def _project_lidar_to_image(
    points_xyz: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    """Project LiDAR XY into image-pixel coordinates.

    The prototype's RGB and LiDAR samples come from different
    surveys with no shared geo-reference, so this projection is a
    *demo-grade* affine fit: scale and centre LiDAR XY into the
    image extent. A production system consumes camera-pose metadata
    from the flight provider and runs a proper ray-cast.

    Documented limitation, not a bug.
    """
    h, w = image_shape
    xy = points_xyz[:, :2].copy()
    xy -= xy.mean(axis=0)
    spread = np.abs(xy).max() + 1e-6
    # Map to roughly the central 70% of the image to leave a margin
    xy = xy * (0.35 * min(h, w) / spread)
    xy[:, 0] += w / 2
    xy[:, 1] += h / 2
    z = points_xyz[:, 2:3] - points_xyz[:, 2:3].min()
    return np.concatenate([xy, z], axis=1).astype(np.float32)


@router.post("/fuse")
@limiter.limit("10/minute")
async def fuse_rgb_lidar(
    request: Request,
    file: UploadFile = File(...),
    lidar_sample: str = "thatcham_sample",
    buffer_px: float = 8.0,
):
    """Run the full RGB → segment → vectorise pipeline, then re-score
    each output LineString with LiDAR-derived features."""
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"file > {settings.max_upload_mb} MB")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}") from e
    if max(image.size) > 4096:
        image.thumbnail((4096, 4096))

    segmenter = request.app.state.segmenter
    prob = segmenter.segment(image)
    skel = mask_to_skeleton(prob, threshold=settings.confidence_threshold)
    graph = skeleton_to_graph(skel)
    lines = graph_to_linestrings(graph, simplify_tol=2.0)

    try:
        lidar_pts, lidar_classes = _load_lidar_sample(lidar_sample)
        lidar_pts_img = _project_lidar_to_image(
            lidar_pts, image_shape=(image.height, image.width)
        )
        lidar_available = True
    except FileNotFoundError:
        # Tab 3's failure mode: no LAZ on disk. Fall back to RGB-only
        # scoring so the endpoint still produces a useful output.
        lidar_pts_img = np.empty((0, 3), dtype=np.float32)
        lidar_classes = None
        lidar_available = False

    fused = fuse(
        linestrings=lines,
        lidar_points=lidar_pts_img,
        lidar_classes=lidar_classes,
        buffer_px=buffer_px,
    )

    def _properties(ls):
        # Look up the matching FusedLineString — order is preserved
        # by `fuse`, so this is just an index lookup.
        f = fused[lines.index(ls)] if lines else None
        if f is None:
            return {}
        return {
            "rgb_length_px": round(f.rgb_length_px, 2),
            "lidar_linearity_support": round(f.lidar_linearity_support, 3),
            "lidar_conductor_fraction": round(f.lidar_conductor_fraction, 3),
            "n_lidar_neighbours": f.n_lidar_neighbours,
            "fused_confidence": round(f.fused_confidence, 3),
        }

    geojson = linestrings_to_geojson(
        lines,
        image_size=(image.height, image.width),
        properties_fn=_properties,
    )

    summary = {
        "n_linestrings": len(lines),
        "lidar_available": lidar_available,
        "lidar_sample": lidar_sample if lidar_available else None,
        "buffer_px": buffer_px,
        "mean_fused_confidence": (
            float(np.mean([f.fused_confidence for f in fused])) if fused else 0.0
        ),
    }
    log.info("fuse.complete", **summary)
    return {"width": image.width, "height": image.height, "geojson": geojson, "summary": summary}
