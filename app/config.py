"""Typed configuration loaded from env / .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model
    model_weights: Path = ROOT / "weights" / "unet_resnet34_ttpla_v2.pth"
    device: str = "cuda"  # falls back to cpu in ConductorSegmenter if unavailable
    tile_size: int = 768
    tile_overlap: int = 64
    confidence_threshold: float = 0.5

    # API
    max_upload_mb: int = 10
    cors_origins: list[str] = ["*"]

    # Examples
    examples_dir: Path = ROOT / "app" / "static" / "examples"

    # Storage (uploaded files retained for N hours, then purged)
    upload_dir: Path = ROOT / "uploads"
    upload_retention_hours: int = 24


settings = Settings()
settings.upload_dir.mkdir(exist_ok=True)
