"""Conductor segmentation model.

Architecture: U-Net with a ResNet34 encoder, ImageNet-pretrained.
Reference: Ronneberger et al. (2015), "U-Net: Convolutional Networks for
Biomedical Image Segmentation" — adapted here for thin-structure (cable)
segmentation in aerial imagery.

The encoder choice trades off depth against the 4 GB VRAM ceiling on the
training laptop (GTX 1050 Ti). ResNet34 sits at the sweet spot: deep enough
to learn powerline texture, shallow enough to fine-tune at 768×768 with
batch size 12 (v2) — the production resolution that matches sliding-window
inference. v1 used 512×512 with batch size 16; the resolution change is
the methodology fix documented in `docs/methodology.md` §10.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import structlog
import torch
from PIL import Image

from app.config import settings

log = structlog.get_logger()


class ConductorSegmenter:
    """Loads a U-Net once, runs inference with sliding-window tiling."""

    def __init__(self, weights_path: Path, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # we load fine-tuned weights below
            in_channels=3,
            classes=1,
        )
        self.weights_path = weights_path
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            # Lightning checkpoints store under 'state_dict'; raw torch.save does not
            if "state_dict" in state:
                state = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
            self.model.load_state_dict(state, strict=False)
            self.weights_loaded = True
            log.info("model.weights_loaded", path=str(weights_path))
        else:
            log.warning(
                "model.weights_missing — running with ImageNet init only",
                path=str(weights_path),
            )
            # Reload with ImageNet so we still produce *something*
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
            self.weights_loaded = False
        self.model.to(self.device).eval()
        self.tile = settings.tile_size
        self.overlap = settings.tile_overlap

    @torch.inference_mode()
    def warmup(self) -> None:
        """One forward pass so the first user request isn't cold."""
        dummy = torch.zeros(1, 3, self.tile, self.tile, device=self.device)
        _ = self.model(dummy)

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> np.ndarray:
        """Run sliding-window inference. Returns float32 mask in [0,1]."""
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        h, w = arr.shape[:2]

        # Pad to multiple of tile
        pad_h = (self.tile - h % self.tile) % self.tile
        pad_w = (self.tile - w % self.tile) % self.tile
        padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        ph, pw = padded.shape[:2]

        prob = np.zeros((ph, pw), dtype=np.float32)
        weight = np.zeros((ph, pw), dtype=np.float32)
        stride = self.tile - self.overlap

        # Hann window suppresses tile-edge artefacts during stitching
        win = np.outer(np.hanning(self.tile), np.hanning(self.tile)).astype(np.float32)

        for y in range(0, ph - self.tile + 1, stride):
            for x in range(0, pw - self.tile + 1, stride):
                tile = padded[y : y + self.tile, x : x + self.tile]
                t = (
                    torch.from_numpy(tile)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self.device, non_blocking=True)
                )
                logits = self.model(t)
                p = torch.sigmoid(logits).squeeze().cpu().numpy()
                prob[y : y + self.tile, x : x + self.tile] += p * win
                weight[y : y + self.tile, x : x + self.tile] += win

        prob /= np.maximum(weight, 1e-6)
        return prob[:h, :w]
