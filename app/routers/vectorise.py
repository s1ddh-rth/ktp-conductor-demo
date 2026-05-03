"""POST /api/vectorise — uploaded image → segmented + vectorised GeoJSON.

Convenience endpoint that runs the full segment → skeleton → graph →
GeoJSON pipeline in one call. The /api/segment endpoint is kept separate
for the case where the frontend wants the raw mask too.
"""
from __future__ import annotations

import base64
import io

import structlog
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from PIL import Image

from app.config import settings
from app.geo.vectorise import linestrings_to_geojson
from app.ml.postprocess import (
    graph_to_linestrings,
    mask_to_skeleton,
    skeleton_to_graph,
)

router = APIRouter()
log = structlog.get_logger()

MAX_BYTES = settings.max_upload_mb * 1024 * 1024


@router.post("/vectorise")
async def vectorise(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Query(
        default=None,
        ge=0.05,
        le=0.95,
        description=(
            "Confidence threshold for binarising the segmenter's probability "
            "map. Defaults to settings.confidence_threshold (0.5). The "
            "frontend slider passes its current value here, so re-thresholding "
            "is a real re-run of the post-processing rather than a UI hint."
        ),
    ),
):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"file > {settings.max_upload_mb} MB")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}") from e

    if max(image.size) > 4096:
        image.thumbnail((4096, 4096))

    seg = request.app.state.segmenter
    prob = seg.segment(image)

    # Resolve threshold: query parameter wins; otherwise the global default.
    thr = threshold if threshold is not None else settings.confidence_threshold

    skel = mask_to_skeleton(prob, threshold=thr)
    graph = skeleton_to_graph(skel)
    lines = graph_to_linestrings(graph, simplify_tol=2.0)
    geojson = linestrings_to_geojson(lines, image_size=(image.height, image.width))

    # Also return the mask for overlay rendering
    import numpy as np

    binary = (prob > thr).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary, mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG", optimize=True)
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    log.info(
        "vectorise.complete",
        threshold=thr,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        n_features=len(geojson["features"]),
    )
    return {
        "width": image.width,
        "height": image.height,
        "mask_png_b64": mask_b64,
        "geojson": geojson,
        "threshold": thr,
        "graph_stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "linestrings": len(geojson["features"]),
        },
    }
