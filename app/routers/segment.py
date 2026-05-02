"""POST /api/segment — RGB image → binary cable mask."""
from __future__ import annotations

import base64
import io

import numpy as np
import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image

from app.config import settings

router = APIRouter()
log = structlog.get_logger()

MAX_BYTES = settings.max_upload_mb * 1024 * 1024


@router.post("/segment")
async def segment(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"file > {settings.max_upload_mb} MB")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}") from e

    # Cap dimensions for laptop GPU
    if max(image.size) > 4096:
        image.thumbnail((4096, 4096))

    segmenter = request.app.state.segmenter
    prob = segmenter.segment(image)

    binary = (prob > settings.confidence_threshold).astype(np.uint8) * 255
    coverage = float((prob > settings.confidence_threshold).mean())
    mean_conf = float(prob[prob > settings.confidence_threshold].mean()) if coverage > 0 else 0.0

    # PNG-encode the mask for the frontend
    mask_img = Image.fromarray(binary, mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG", optimize=True)
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    log.info(
        "segment.complete",
        size=image.size,
        coverage=round(coverage, 4),
        mean_conf=round(mean_conf, 3),
    )
    return {
        "width": image.width,
        "height": image.height,
        "mask_png_b64": mask_b64,
        "metrics": {
            "coverage": coverage,
            "mean_confidence": mean_conf,
            "threshold": settings.confidence_threshold,
        },
    }
