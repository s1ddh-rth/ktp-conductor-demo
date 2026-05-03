"""FastAPI entrypoint for the KTP conductor-detection demo.

Pipeline overview:
  /api/segment        RGB image  →  binary mask
  /api/vectorise      mask       →  GeoJSON linestrings
  /api/infer-hidden   partial graph + poles → completed cable network
  /api/lidar          LAS subset →  per-point classification

The application is intentionally a single uvicorn process. The model is
loaded once at startup and held in GPU memory; all endpoints are stateless
beyond that.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.ml.model import ConductorSegmenter
from app.routers import fuse, infer, lidar, segment, vectorise

# ── logging ────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

# ── lifespan: load model on startup, free GPU on shutdown ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup.begin", weights=str(settings.model_weights))
    app.state.segmenter = ConductorSegmenter(
        weights_path=settings.model_weights,
        device=settings.device,
    )
    app.state.segmenter.warmup()
    log.info("startup.ready", device=app.state.segmenter.device)
    yield
    log.info("shutdown.begin")
    del app.state.segmenter
    log.info("shutdown.complete")


# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KTP Conductor Demo",
    description=(
        "Aerial conductor segmentation, vectorisation, hidden-cable "
        "inference, and LiDAR classification."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limit: protect the laptop from a viral link
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — relaxed for the demo subdomain; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ── per-request structured logging ─────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", f"r{int(time.time() * 1000)}")
    start = time.perf_counter()
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("request", status=response.status_code, ms=round(elapsed_ms, 1))
        return response
    finally:
        structlog.contextvars.clear_contextvars()


# ── error handler: never leak stack traces ─────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    log.exception("unhandled", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "see server logs"},
    )


# ── health ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health(request: Request):
    seg = request.app.state.segmenter
    return {
        "status": "ok",
        "model_loaded": seg is not None,
        "device": seg.device if seg else None,
    }


# ── routers ────────────────────────────────────────────────────────────────
app.include_router(segment.router, prefix="/api", tags=["segment"])
app.include_router(vectorise.router, prefix="/api", tags=["vectorise"])
app.include_router(infer.router, prefix="/api", tags=["infer-hidden"])
app.include_router(lidar.router, prefix="/api", tags=["lidar"])
app.include_router(fuse.router, prefix="/api", tags=["fuse"])

# ── static SPA at / ────────────────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
