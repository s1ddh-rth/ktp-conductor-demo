"""LiDAR per-point classification using eigenvalue features.

Reference
---------
Demantké, J., Mallet, C., David, N., & Vallet, B. (2011).
"Dimensionality based scale selection in 3D lidar point clouds."
ISPRS Workshop on Laser Scanning.

For each point, compute the local covariance matrix on a k-nearest-
neighbour ball; its sorted eigenvalues λ1 ≥ λ2 ≥ λ3 give shape descriptors:

    linearity   = (λ1 − λ2) / λ1     (high for cables/wires)
    planarity   = (λ2 − λ3) / λ1     (high for ground/roof)
    sphericity  = λ3 / λ1            (high for vegetation)
    anisotropy  = (λ1 − λ3) / λ1
    verticality = 1 − |n_z|          (high for poles)

Combined with height-above-ground, this is enough to label a point cloud
into {ground, vegetation, conductor, structure} for a clean visual demo.
No deep learning required — and the interpretability is itself a feature.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_features(points: np.ndarray, k: int = 16) -> dict[str, np.ndarray]:
    """k-NN eigenvalue features per point."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)  # +1 because query includes self
    idx = idx[:, 1:]
    neighbours = points[idx]  # (N, k, 3)
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / k

    # Eigendecomp; numpy returns ascending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[:, ::-1]  # descending
    eigvecs = eigvecs[:, :, ::-1]
    eigvals = np.clip(eigvals, 1e-9, None)

    l1, l2, l3 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1
    # Normal is the eigenvector associated with the smallest eigenvalue
    normals = eigvecs[:, :, 2]
    verticality = 1.0 - np.abs(normals[:, 2])

    return {
        "linearity": linearity,
        "planarity": planarity,
        "sphericity": sphericity,
        "verticality": verticality,
    }


def classify(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Return integer class per point.

    Classes
    -------
    0  ground
    1  vegetation
    2  conductor (powerline)
    3  structure (pole / building)
    """
    feat = compute_features(points, k=k)
    z = points[:, 2]
    z_ground = np.percentile(z, 5)  # crude ground estimate
    height = z - z_ground

    classes = np.zeros(len(points), dtype=np.int8)

    # Vegetation: low planarity, high sphericity, mid-height
    veg = (feat["sphericity"] > 0.15) & (height > 1.0) & (height < 25.0)
    classes[veg] = 1

    # Conductor: high linearity, elevated, low vertical extent
    cond = (feat["linearity"] > 0.85) & (height > 3.0) & (feat["verticality"] < 0.3)
    classes[cond] = 2

    # Structure: high planarity OR high verticality, elevated
    struct = ((feat["planarity"] > 0.5) | (feat["verticality"] > 0.7)) & (height > 1.5)
    struct &= ~cond  # don't overwrite cables
    classes[struct] = 3

    return classes
