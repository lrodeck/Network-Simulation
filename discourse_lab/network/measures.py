"""Small graph diagnostics used by the step-4 verification tests (dev §6)."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def degree_sequence(G: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(G.sum(axis=1)).ravel()


def global_clustering(G: sparse.csr_matrix) -> float:
    """Global (transitivity) clustering coefficient on the undirected version."""
    A = (G + G.T) > 0
    A = A.astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    triangles = (A @ A @ A).diagonal().sum() / 6.0
    possible = (deg * (deg - 1)).sum() / 2.0
    if possible == 0:
        return 0.0
    return float(3 * triangles / possible)


def mean_neighbor_distance(G: sparse.csr_matrix, coords: np.ndarray) -> float:
    """Mean latent-space distance between an edge's two endpoints — a
    homophily proxy: lower means neighbourhoods are more alike than chance.
    """
    coo = G.tocoo()
    if coo.nnz == 0:
        return float("nan")
    d = np.linalg.norm(coords[coo.row] - coords[coo.col], axis=1)
    return float(d.mean())
