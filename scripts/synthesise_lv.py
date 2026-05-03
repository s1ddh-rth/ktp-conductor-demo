"""Synthesise urban-LV-style aerial scenes for qualitative demonstration.

Why this exists
---------------
The biggest credibility gap in the prototype is that TTPLA is *rural
transmission lines*, not *urban LV service cables*. We can narrow the
visual gap (not the model accuracy gap) by generating a few synthetic
urban-LV scenes: a procedural top-down view of a housing estate with
catenary cables strung between hand-placed pole positions.

The geometry of each cable is computed by the production
``fit_catenary_2d`` (``app/ml/catenary.py``), so the synthesis and the
catenary endpoint of the demo share their physical model. The cable
strokes are anti-aliased with PIL so the result looks photographic
rather than vector-traced.

Output
------
Three image/mask pairs at ``app/static/examples/synthetic_lv_{1..3}.jpg``
and ``synthetic_lv_{1..3}_mask.png``. The mask is binary (0/255) and
matches the rasterised cable centerlines plus a small dilation, suitable
as a ground-truth fixture for a qualitative pass through the segmenter.

Caveat
------
This is a **methodology demonstrator**, not a training augmentation.
Real synthetic-data augmentation for LV segmentation needs a far richer
generator — varied lighting, realistic occluders, true shadow casting,
and a sampling distribution that reflects real urban roof geometry.
Madaan, Maturana, Scherer (2017), *Wire Detection using Synthetic Data
and Dilated Convolutional Networks for Unmanned Aerial Vehicles*, is
the canonical starting reference for that direction.

Usage
-----
    python -m scripts.synthesise_lv \\
        --out app/static/examples \\
        --n 3 \\
        --size 768

The default seed (42) yields three deterministic scenes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.ml.catenary import fit_catenary_2d


# ── procedural backgrounds ─────────────────────────────────────────────
@dataclass
class Scene:
    """One synthetic scene's parameters: layout + cable plan."""

    seed: int
    poles: list[tuple[float, float]]
    cables: list[tuple[int, int]]  # (i, j) pole indices to connect
    sag_fraction: float
    label: str


def _random_aerial_background(rng: np.random.Generator, size: int) -> Image.Image:
    """A coarse top-down aerial-style background.

    Composes:
    - Soft greyscale Perlin-like noise standing in for ground texture.
    - Rectangular 'rooftop' patches in muted browns and greys.
    - Thin tarmac strip standing in for a road.

    No real imagery is downloaded; this avoids licensing concerns and
    keeps the synthesis deterministic offline. To swap in a real OSM
    or OpenAerialMap tile, replace this function — the rest of the
    pipeline operates on any RGB ``Image`` of the requested size.
    """
    h = w = size
    # Soft multi-scale noise for the ground
    base = np.zeros((h, w), dtype=np.float32)
    for scale in (4, 16, 64):
        coarse = rng.normal(0, 1, size=(h // scale + 2, w // scale + 2)).astype(np.float32)
        layer = np.array(
            Image.fromarray(coarse).resize((w, h), Image.BICUBIC), dtype=np.float32
        )
        base += layer / scale
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    # Brighter ground so the dark cables sit on a higher-contrast
    # background than they did in v1 — bumps cables into the
    # detector's response range without losing the aerial-photo feel.
    ground = (115 + 55 * base).astype(np.uint8)
    rgb = np.stack([ground, ground - 8, ground - 18], axis=-1).clip(60, 220).astype(np.uint8)
    img = Image.fromarray(rgb)

    draw = ImageDraw.Draw(img)
    # Tarmac road horizontally across the lower third
    road_y = int(0.66 * h + rng.uniform(-0.05, 0.05) * h)
    draw.rectangle((0, road_y - 18, w, road_y + 18), fill=(58, 60, 64))
    # Faint lane markings
    for x in range(40, w, 80):
        draw.rectangle((x, road_y - 1, x + 28, road_y + 1), fill=(220, 220, 200))

    # 8–14 rectangular rooftops
    palette = [
        (140, 90, 70), (170, 110, 90), (110, 100, 95),
        (160, 140, 110), (90, 80, 75), (180, 160, 130),
    ]
    n_roofs = int(rng.integers(8, 15))
    for _ in range(n_roofs):
        rw = int(rng.integers(48, 110))
        rh = int(rng.integers(36, 80))
        rx = int(rng.integers(8, w - rw - 8))
        ry = int(rng.integers(8, road_y - rh - 12))
        colour = palette[int(rng.integers(0, len(palette)))]
        draw.rectangle((rx, ry, rx + rw, ry + rh), fill=colour)
        # Roof ridge (subtle highlight)
        draw.line(
            (rx + 4, ry + rh // 2, rx + rw - 4, ry + rh // 2),
            fill=tuple(min(255, c + 25) for c in colour),
            width=2,
        )

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


# ── catenary rasteriser ────────────────────────────────────────────────
def _draw_cable(
    base: Image.Image,
    mask: Image.Image,
    p1: tuple[float, float],
    p2: tuple[float, float],
    sag_fraction: float,
    rng: np.random.Generator,
) -> None:
    """Render a catenary cable on `base` and burn its centerline to `mask`.

    The cable is drawn as a thin anti-aliased line (PIL's `line` with
    width=1 is hard-edged; we draw on a 4× super-sampled buffer and
    downsample for a smoother appearance). The mask burns the
    centerline path at the *output* resolution, dilated by 1 pixel —
    matching the typical thickness of a real LV service cable in
    aerial imagery (~30 mm cable at 5 cm GSD ≈ 1 px).
    """
    curve = fit_catenary_2d(p1, p2, sag_fraction=sag_fraction, n_points=80)
    # Cable colour: very dark grey with slight chromatic jitter.
    # Tightened range (10–25) for higher contrast against the brighter
    # ground — the v1 (20–45) range was below the TTPLA-trained
    # detector's edge-response threshold.
    rgb = (
        int(rng.integers(10, 25)),
        int(rng.integers(10, 25)),
        int(rng.integers(10, 25)),
    )

    # Super-sampled image draw
    scale = 4
    w, h = base.size
    super_img = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    sd = ImageDraw.Draw(super_img)
    pts_super = [(x * scale, y * scale) for x, y in curve]
    # Effective output stroke width ≈ 2.5 px after the 4× downsample,
    # matching TTPLA's typical cable-on-imagery width of 2-4 px.
    sd.line(pts_super, fill=(*rgb, 240), width=int(2.5 * scale), joint="curve")
    super_img = super_img.resize((w, h), Image.LANCZOS)
    base.alpha_composite(super_img)

    # Mask: integer-quantised pixel walk along the curve
    md = ImageDraw.Draw(mask)
    md.line([(float(x), float(y)) for x, y in curve], fill=255, width=2)


def _draw_pole(image: Image.Image, x: float, y: float, rng: np.random.Generator) -> None:
    """Tiny grey square standing in for a pole top, with shadow."""
    s = int(rng.integers(3, 5))
    d = ImageDraw.Draw(image)
    d.ellipse((x - s - 1, y - s + 1, x + s + 1, y + s + 3), fill=(0, 0, 0, 110))
    d.ellipse((x - s, y - s, x + s, y + s), fill=(60, 50, 40, 240))


# ── scene definition ───────────────────────────────────────────────────
def _make_scenes(size: int, seed: int = 42) -> list[Scene]:
    """Return three deterministic scene plans.

    Pole positions are chosen to give visually distinct topologies:

    1. Two parallel feeders along a street.
    2. A simple star from a transformer pole to four houses.
    3. A long single-feeder run across the scene with multiple drops.
    """
    s = size
    return [
        Scene(
            seed=seed,
            label="parallel_street_run",
            poles=[
                (0.10 * s, 0.55 * s),
                (0.32 * s, 0.55 * s),
                (0.55 * s, 0.55 * s),
                (0.78 * s, 0.55 * s),
                (0.93 * s, 0.55 * s),
            ],
            cables=[(0, 1), (1, 2), (2, 3), (3, 4)],
            sag_fraction=0.025,
        ),
        Scene(
            seed=seed + 1,
            label="transformer_star",
            poles=[
                (0.50 * s, 0.50 * s),  # transformer pole (centre)
                (0.20 * s, 0.30 * s),
                (0.80 * s, 0.30 * s),
                (0.20 * s, 0.78 * s),
                (0.80 * s, 0.78 * s),
            ],
            cables=[(0, 1), (0, 2), (0, 3), (0, 4)],
            sag_fraction=0.04,
        ),
        Scene(
            seed=seed + 2,
            label="long_feeder_with_drops",
            poles=[
                (0.05 * s, 0.62 * s),
                (0.28 * s, 0.62 * s),
                (0.50 * s, 0.62 * s),
                (0.72 * s, 0.62 * s),
                (0.95 * s, 0.62 * s),
                (0.28 * s, 0.30 * s),
                (0.50 * s, 0.20 * s),
                (0.72 * s, 0.30 * s),
            ],
            cables=[
                (0, 1), (1, 2), (2, 3), (3, 4),  # main run
                (1, 5), (2, 6), (3, 7),          # drops to houses
            ],
            sag_fraction=0.03,
        ),
    ]


def synthesise_one(scene: Scene, size: int) -> tuple[Image.Image, Image.Image]:
    """Render one (image, mask) pair for the given scene."""
    rng = np.random.default_rng(scene.seed)
    bg = _random_aerial_background(rng, size).convert("RGBA")
    mask = Image.new("L", (size, size), 0)

    for i, j in scene.cables:
        _draw_cable(bg, mask, scene.poles[i], scene.poles[j], scene.sag_fraction, rng)

    for x, y in scene.poles:
        _draw_pole(bg, x, y, rng)

    return bg.convert("RGB"), mask


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path("app/static/examples"))
    p.add_argument("--n", type=int, default=3, help="number of scenes (capped at 3)")
    p.add_argument("--size", type=int, default=768, help="square output side in pixels")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scenes = _make_scenes(args.size, seed=args.seed)[: args.n]

    for i, scene in enumerate(scenes, 1):
        img, mask = synthesise_one(scene, args.size)
        img_path = args.out / f"synthetic_lv_{i:02d}.jpg"
        mask_path = args.out / f"synthetic_lv_{i:02d}_mask.png"
        img.save(img_path, quality=92)
        mask.save(mask_path, optimize=True)
        cov = float((np.array(mask) > 127).mean())
        print(
            f"scene {i} ({scene.label}): {img_path.name}  "
            f"cable pixel fraction = {cov*100:.2f}%"
        )

    print(
        f"\nWrote {len(scenes)} pairs to {args.out}. "
        "Add them to app/static/examples/index.json to surface in the UI."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
