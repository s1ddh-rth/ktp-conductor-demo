"""POST /api/infer-hidden — completion methods for occluded conductors.

Two modes:
  - 'catenary': fit a catenary curve between two anchor points.
  - 'topology': infer the LV cable network connecting buildings to a
    transformer, optionally biased by observed cable fragments.

Inputs are JSON to keep this endpoint fast and exemplar-driven; the
frontend ships pre-defined scenarios that demonstrate the methods.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from shapely.geometry import LineString

from app.limiter import limiter
from app.ml.catenary import confidence_band, fit_catenary_2d
from app.ml.graph_complete import confidence_per_edge, predict_lv_topology

router = APIRouter()
log = structlog.get_logger()


class CatenaryRequest(BaseModel):
    p1: tuple[float, float]
    p2: tuple[float, float]
    sag_fraction: float = 0.02


class TopologyRequest(BaseModel):
    transformer: tuple[float, float]
    buildings: list[tuple[float, float]]
    observed_fragments: list[list[tuple[float, float]]] = []
    pole_candidates: list[tuple[float, float]] = []


@router.post("/infer-hidden/catenary")
@limiter.limit("30/minute")
async def infer_catenary(request: Request, req: CatenaryRequest):
    if req.p1 == req.p2:
        raise HTTPException(400, "p1 and p2 must differ")
    curve = fit_catenary_2d(req.p1, req.p2, sag_fraction=req.sag_fraction)
    upper, lower = confidence_band(curve)
    return {
        "curve": curve.tolist(),
        "band": {"upper": upper.tolist(), "lower": lower.tolist()},
    }


@router.post("/infer-hidden/topology")
@limiter.limit("30/minute")
async def infer_topology(request: Request, req: TopologyRequest):
    fragments = [LineString(f) for f in req.observed_fragments if len(f) >= 2]
    edges = predict_lv_topology(
        transformer=req.transformer,
        buildings=req.buildings,
        observed_fragments=fragments,
        pole_candidates=req.pole_candidates,
    )
    output = []
    for ls in edges:
        coords = list(ls.coords)
        conf = confidence_per_edge(ls, fragments) if fragments else 0.0
        output.append({"coords": coords, "evidence_support": conf})
    log.info("topology.complete", n_edges=len(output))
    return {"edges": output, "n_edges": len(output)}
