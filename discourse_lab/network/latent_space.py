"""Latent-space graph generator with preferential attachment (spec §2.2).

    logit P(u -> v) = alpha - beta * d(u, v) + gamma * log(1 + prominence_v)
    d(u, v) = || [s_u; a_u] - [s_v; a_v] ||_2

`alpha` is calibrated by bisection to hit the target mean degree. Candidate
edges come from a kNN search in latent space (exact for small N, and the only
tractable option once N grows past ~20k) plus a uniform long-tie component —
a pure kNN graph has no shortcuts and cascades die in-cluster.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import expit

from discourse_lab.config import Config
from discourse_lab.population import Population
from discourse_lab.registry import register


def _latent_coords(pop: Population) -> np.ndarray:
    cols = [i for i, name in enumerate(pop.trait_names) if name.startswith("stance_") or name.startswith("topic_affinity_")]
    return pop.X_used[:, cols]


def _candidate_edges(pop: Population, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """kNN candidate pool: (src, dst, latent_distance), self excluded."""
    L = _latent_coords(pop)
    n = L.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0)
    tree = cKDTree(L)
    dists, idxs = tree.query(L, k=k + 1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]  # drop self (distance 0 at column 0)
    src = np.repeat(np.arange(n), k)
    return src, idxs.reshape(-1), dists.reshape(-1)


def _calibrate_alpha(
    logit_base: np.ndarray, target_mean_degree: float, n: int, rng: np.random.Generator, iterations: int = 40
) -> float:
    """Bisection on alpha so the expected candidate-edge count hits the target
    mean degree (expectation, not a single stochastic draw, so it is stable).
    """
    lo, hi = -30.0, 30.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        expected_degree = expit(logit_base + mid).sum() / n
        if expected_degree < target_mean_degree:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@register("graph_generator", "latent_space")
def latent_space_graph(cfg: Config, pop: Population, rng: np.random.Generator) -> sparse.csr_matrix:
    gcfg = cfg.graph
    n = pop.X_used.shape[0]
    prominence = pop.X_used[:, pop.trait_names.index("prominence")]
    log_prom = np.log1p(prominence)

    src, dst, dist = _candidate_edges(pop, gcfg.knn_k)
    logit_base = -gcfg.homophily_beta * dist + gcfg.prominence_gamma * log_prom[dst]

    alpha = _calibrate_alpha(logit_base, gcfg.mean_degree, n, rng)
    probs = expit(logit_base + alpha)
    draws = rng.random(probs.shape) < probs

    rows, cols = src[draws], dst[draws]

    n_long = int(gcfg.long_tie_fraction * len(rows))
    if n_long > 0:
        long_rows = rng.integers(0, n, n_long)
        long_cols = rng.integers(0, n, n_long)
        keep = long_rows != long_cols
        rows = np.concatenate([rows, long_rows[keep]])
        cols = np.concatenate([cols, long_cols[keep]])

    data = np.ones(len(rows), dtype=np.int8)
    G = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    G.setdiag(0)
    G.eliminate_zeros()
    G.data[:] = 1
    return G
