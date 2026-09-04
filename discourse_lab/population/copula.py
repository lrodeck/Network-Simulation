"""Gaussian copula latent + Higham nearest-PSD correlation (spec §2.1).

Correlations are specified sparsely as `(trait_i, trait_j, rho)` pairs and
completed to a valid correlation matrix by nearest-PSD projection — hand
authoring a full n×n matrix is not guaranteed PSD and the failure is silent.
"""

from __future__ import annotations

import numpy as np


def sparse_pairs_to_matrix(
    trait_names: list[str], pairs: tuple[tuple[str, str, float], ...]
) -> np.ndarray:
    n = len(trait_names)
    idx = {name: i for i, name in enumerate(trait_names)}
    corr = np.eye(n)
    for a, b, rho in pairs:
        i, j = idx[a], idx[b]
        corr[i, j] = rho
        corr[j, i] = rho
    return corr


def nearest_psd_correlation(corr: np.ndarray, iterations: int = 100, tol: float = 1e-10) -> np.ndarray:
    """Higham (2002) alternating projections onto the PSD cone and the set of
    unit-diagonal ("correlation") matrices.
    """
    n = corr.shape[0]
    y = corr.copy()
    delta_s = np.zeros_like(corr)
    prev = np.zeros_like(corr)

    for _ in range(iterations):
        r = y - delta_s
        eigval, eigvec = np.linalg.eigh((r + r.T) / 2)
        eigval_clipped = np.clip(eigval, 0, None)
        x = (eigvec * eigval_clipped) @ eigvec.T
        delta_s = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0)

        if np.linalg.norm(y - prev, ord="fro") < tol:
            break
        prev = y.copy()

    y = (y + y.T) / 2
    np.fill_diagonal(y, 1.0)
    return y


def mixture_moments(weights: np.ndarray, means: np.ndarray, var_diag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of a Gaussian mixture with shared covariance diagonal
    `var_diag` and per-component means `means` (C x n), by law of total
    variance: Var = E[Var|c] + Var[E[·|c]].
    """
    mu_bar = weights @ means
    between = weights @ (means - mu_bar) ** 2
    var_bar = var_diag + between
    return mu_bar, np.sqrt(var_bar)


def sample_latent(
    rng: np.random.Generator,
    weights: np.ndarray,
    component_means: np.ndarray,
    corr: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw archetype labels then correlated latent z (spec §2.1: `z_u ~
    N(mu_{c_u}, Sigma)`, shared correlation across components — offsets shift
    the mean only). Returns (z, labels).
    """
    labels = rng.choice(len(weights), size=n, p=weights)
    chol = np.linalg.cholesky(corr)
    noise = rng.standard_normal((n, corr.shape[0])) @ chol.T
    z = component_means[labels] + noise
    return z, labels
