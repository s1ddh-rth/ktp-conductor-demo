"""Generate a deterministic synthetic LiDAR tile for the LiDAR demo tab.

Why this exists
---------------
The LiDAR tab in the demo expects a `.laz` file at
``app/static/examples/thatcham_sample.laz``. In a real deployment
that file comes from the Environment Agency LiDAR portal or an
equivalent source. For local development and the public-repo demo,
a curated real tile is awkward to ship (file size, licensing,
geo-reference clutter), so we generate a small synthetic point
cloud with the same structural elements the classifier handles:

- a noisy ground plane
- a few vertical poles
- horizontal cable lines spanning between poles
- low-rise vegetation clusters

The synthesiser is deterministic (seed-driven) so the demo's LiDAR
view is identical across runs.

Limits made explicit
--------------------
This is a *demo fixture*, not training data. The cable runs are
straight horizontal lines, not catenaries; the vegetation is
spherical noise, not realistic canopy. For training a learned
LiDAR classifier (per the research roadmap) the synthesiser would
need to match real Environment Agency tile statistics — point
density, scan angle distribution, intensity returns.

Usage
-----
    python -m scripts.synthesise_lidar \\
        --out app/static/examples/thatcham_sample.laz

The ``app.routers.lidar`` and ``app.routers.fuse`` endpoints both
pick up the resulting file automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import laspy
import numpy as np


def _ground(rng: np.random.Generator, side: float, density: float) -> np.ndarray:
    """A noisy roughly-flat ground plane."""
    n = int(side * side * density)
    xy = rng.uniform(-side / 2, side / 2, size=(n, 2))
    # Gentle undulation + small Gaussian noise
    z = 0.05 * np.sin(0.05 * xy[:, 0]) + 0.1 * rng.standard_normal(n)
    return np.column_stack([xy, z]).astype(np.float32)


def _pole(
    rng: np.random.Generator,
    centre: tuple[float, float],
    height: float,
    n: int = 80,
) -> np.ndarray:
    """A vertical pole — eigenvalue features should report it as
    high-linearity AND high-verticality."""
    x, y = centre
    z = np.linspace(0, height, n)
    pts = np.column_stack(
        [np.full(n, x), np.full(n, y), z]
    ) + 0.02 * rng.standard_normal((n, 3))
    return pts.astype(np.float32)


def _cable(
    rng: np.random.Generator,
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    n: int = 200,
) -> np.ndarray:
    """A horizontal cable run between two anchor points.

    Real cables follow a catenary, but for the demo's eigenvalue
    classifier the linear-vs-not distinction is what matters; a
    straight-line approximation is sufficient.
    """
    t = np.linspace(0, 1, n)
    pts = np.outer(1 - t, p1) + np.outer(t, p2)
    pts += 0.03 * rng.standard_normal(pts.shape)
    return pts.astype(np.float32)


def _vegetation(
    rng: np.random.Generator,
    centre: tuple[float, float],
    radius: float,
    n: int = 600,
) -> np.ndarray:
    """A roughly-spherical vegetation blob — high sphericity per
    Demantké et al. (2011)."""
    x, y = centre
    pts = rng.normal(0, radius, size=(n, 3))
    pts[:, 0] += x
    pts[:, 1] += y
    pts[:, 2] = np.abs(pts[:, 2]) + 1.5  # vegetation lives above ground
    return pts.astype(np.float32)


def synthesise(side: float = 80.0, seed: int = 42) -> np.ndarray:
    """Compose a single synthetic tile.

    Layout: a row of three poles along the y=0 axis with cables
    running between them, two vegetation clusters off to the sides,
    and a ground plane underneath. ~25k points after composition.
    """
    rng = np.random.default_rng(seed)

    pieces: list[np.ndarray] = []
    pieces.append(_ground(rng, side, density=2.0))

    pole_x = [-25.0, 0.0, 25.0]
    pole_height = 9.0
    for x in pole_x:
        pieces.append(_pole(rng, centre=(x, 0.0), height=pole_height))

    # Cables strung between consecutive poles, slightly drooping
    for i in range(len(pole_x) - 1):
        pieces.append(
            _cable(
                rng,
                p1=(pole_x[i], 0.0, pole_height - 0.5),
                p2=(pole_x[i + 1], 0.0, pole_height - 0.5),
                n=220,
            )
        )

    # Off-axis vegetation
    pieces.append(_vegetation(rng, centre=(-15.0, 18.0), radius=4.0))
    pieces.append(_vegetation(rng, centre=(20.0, -22.0), radius=5.5))

    return np.concatenate(pieces, axis=0)


def write_laz(points: np.ndarray, out_path: Path) -> None:
    """Write a minimal LAZ file (point format 1) compatible with
    laspy and the demo's existing reader."""
    header = laspy.LasHeader(point_format=1, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = points.min(axis=0)
    las = laspy.LasData(header=header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(out_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--out",
        type=Path,
        default=Path("app/static/examples/thatcham_sample.laz"),
        help="output .laz path",
    )
    p.add_argument("--side", type=float, default=80.0, help="tile side in metres")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pts = synthesise(side=args.side, seed=args.seed)
    write_laz(pts, args.out)
    print(f"Wrote {len(pts):,} points to {args.out}")
    print(f"  bounds: x[{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}]  "
          f"y[{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]  "
          f"z[{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
