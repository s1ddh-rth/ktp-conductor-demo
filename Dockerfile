# Reproducible runtime for the KTP conductor-detection demo.
#
# Stage 1 builds the venv with `uv sync` so the wheel cache lives in a
# disposable layer; stage 2 is a slim runtime that copies just the venv
# and the app.
#
# CUDA 12.4 runtime matches the PyTorch 2.4 wheel we depend on (ROCm
# users should swap the base image for `rocm/pytorch:latest` and rerun
# `uv sync`). The :runtime variant lacks nvcc, which we don't need at
# inference time.

ARG CUDA_TAG=12.4.1-runtime-ubuntu22.04


# ── stage 1: build venv with uv ────────────────────────────────────────
FROM nvidia/cuda:${CUDA_TAG} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        ca-certificates curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv is shipped as a single static binary; no Python install required.
COPY --from=ghcr.io/astral-sh/uv:0.4.27 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
COPY app/ ./app/
COPY training/ ./training/
COPY scripts/ ./scripts/

# `--frozen` is omitted because uv.lock may not be present on first
# build; switch it on once the lockfile is committed.
RUN uv sync --no-dev


# ── stage 2: runtime ──────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_TAG}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 ca-certificates curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 ktp

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=ktp:ktp app/ ./app/
COPY --chown=ktp:ktp training/ ./training/
COPY --chown=ktp:ktp scripts/ ./scripts/

# Mount points for runtime artefacts the user provides
RUN install -d -o ktp -g ktp /app/weights /app/uploads
USER ktp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
