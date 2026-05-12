"""POST /api/lidar — load and classify a LAS subset.

For the demo, the LAS file lives on the server (curated sample). The
endpoint subsamples to a browser-friendly point count, classifies, and
returns coordinates + classes for Three.js rendering.
"""
from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import structlog
from fastapi import APIRouter, HTTPException, Query, Request

from app.config import settings
from app.geo.lidar_features import classify
from app.limiter import limiter

router = APIRouter()
log = structlog.get_logger()


@router.get("/lidar/sample")
@limiter.limit("20/minute")
async def lidar_sample(
    request: Request,
    name: str = Query("thatcham_sample", description="sample LAS basename"),
    max_points: int = Query(150_000, ge=10_000, le=500_000),
):
    """Return a classified point cloud subset for browser rendering."""
    path = settings.examples_dir / f"{name}.laz"
    if not path.exists():
        path = settings.examples_dir / f"{name}.las"
    if not path.exists():
        raise HTTPException(404, f"sample '{name}' not found")

    with laspy.open(path) as fh:
        las = fh.read()

    pts = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
    if len(pts) > max_points:
        # Seeded so Tab 3's view and /api/fuse's LiDAR subsampling agree.
        idx = np.random.default_rng(0).choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]

    # Re-centre for browser numerical stability
    centroid = pts.mean(axis=0)
    pts -= centroid

    classes = classify(pts, k=16)
    log.info(
        "lidar.classified",
        n=len(pts),
        ground=int((classes == 0).sum()),
        veg=int((classes == 1).sum()),
        cond=int((classes == 2).sum()),
        struct=int((classes == 3).sum()),
    )
    return {
        "n_points": int(len(pts)),
        "centroid": centroid.tolist(),
        "points": pts.flatten().tolist(),  # x0,y0,z0, x1,y1,z1, …
        "classes": classes.tolist(),
        "class_names": ["ground", "vegetation", "conductor", "structure"],
    }
