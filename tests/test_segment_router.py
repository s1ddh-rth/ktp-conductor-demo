"""Smoke tests for the FastAPI segment router.

These exercise the request-shape contract. The actual model output is
non-deterministic (depends on whether weights are present), so the
assertions are deliberately lenient: we check that 200 returns include
the documented fields with correct types and that the mask PNG decodes
to the requested image dimensions.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Drive lifespan startup so the segmenter is loaded once."""
    with TestClient(app) as c:
        yield c


def _make_png_bytes(size: tuple[int, int] = (128, 128)) -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint_reports_model_state(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "model_loaded" in body
    assert "device" in body


def test_segment_returns_mask_with_correct_dimensions(client):
    png = _make_png_bytes((128, 128))
    r = client.post(
        "/api/segment",
        files={"file": ("test.png", png, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 128
    assert body["height"] == 128
    decoded = Image.open(io.BytesIO(base64.b64decode(body["mask_png_b64"])))
    assert decoded.size == (128, 128)
    metrics = body["metrics"]
    assert {"coverage", "mean_confidence", "threshold"} <= metrics.keys()
    assert 0.0 <= metrics["coverage"] <= 1.0


def test_segment_rejects_invalid_image(client):
    r = client.post(
        "/api/segment",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400


def test_vectorise_returns_geojson_feature_collection(client):
    png = _make_png_bytes((96, 96))
    r = client.post(
        "/api/vectorise",
        files={"file": ("test.png", png, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["geojson"]["type"] == "FeatureCollection"
    assert isinstance(body["geojson"]["features"], list)
    assert {"nodes", "edges", "linestrings"} <= body["graph_stats"].keys()


def test_catenary_endpoint_returns_curve(client):
    r = client.post(
        "/api/infer-hidden/catenary",
        json={"p1": [0.0, 0.0], "p2": [100.0, 0.0]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["curve"]) >= 2
    assert "upper" in body["band"]
    assert "lower" in body["band"]


def test_catenary_endpoint_rejects_equal_anchors(client):
    r = client.post(
        "/api/infer-hidden/catenary",
        json={"p1": [10.0, 10.0], "p2": [10.0, 10.0]},
    )
    assert r.status_code == 400


def test_topology_endpoint_returns_edges(client):
    r = client.post(
        "/api/infer-hidden/topology",
        json={
            "transformer": [0.0, 0.0],
            "buildings": [[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]],
            "observed_fragments": [],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_edges"] >= 3  # at least one per building
