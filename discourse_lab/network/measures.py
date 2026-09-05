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


def reciprocity(G: sparse.csr_matrix) -> float:
    """Share of edges whose reverse edge also exists (spec §5.1: 0.2-0.4).

    `network/reciprocity.py` only *generates* reciprocal edges; nothing
    measured the result until now, so the target range was unverifiable.
    """
    A = (G > 0).astype(np.int8)
    if A.nnz == 0:
        return float("nan")
    return float(A.multiply(A.T).nnz / A.nnz)


def configuration_null(
    G: sparse.csr_matrix, rng: np.random.Generator, swaps_per_edge: int = 5
) -> sparse.csr_matrix:
    """A degree-preserving null: exactly the same in- and out-degree sequence
    as `G`, rewired at random by double-edge swaps.

    Two things this is deliberately *not*:

    - It is not the `configuration_model` generator in
      `network/configuration.py`. That draws `Poisson(cfg.graph.mean_degree)`
      out-degrees and picks destinations by prominence, so it matches a mean
      degree and a prominence tail, not the observed sequence — comparing
      clustering against it would confound rewiring with a different degree
      distribution.
    - It is not stub-shuffling. Shuffling destination stubs is simpler, but
      collapsing the duplicate edges it creates loses ~8% of edges at this
      graph's density, which *lowers* the null's clustering and so inflates
      the observed/null ratio in the model's favour. Double-edge swaps keep
      every degree exactly and the graph simple.
    """
    A = (G > 0).tocoo()
    n = G.shape[0]
    if A.nnz == 0:
        return sparse.csr_matrix((n, n), dtype=np.int8)

    edges = list(zip(A.row.tolist(), A.col.tolist()))
    edge_set = set(edges)
    m = len(edges)

    attempts = swaps_per_edge * m
    picks = rng.integers(0, m, size=(attempts, 2))
    for i, j in picks:
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if a == d or c == b:
            continue  # would be a self-loop
        if (a, d) in edge_set or (c, b) in edge_set:
            continue  # would duplicate an existing edge
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add((a, d))
        edge_set.add((c, b))
        edges[i] = (a, d)
        edges[j] = (c, b)

    rows = np.fromiter((e[0] for e in edges), dtype=np.int64, count=m)
    cols = np.fromiter((e[1] for e in edges), dtype=np.int64, count=m)
    null = sparse.coo_matrix((np.ones(m, dtype=np.int8), (rows, cols)), shape=(n, n)).tocsr()
    null.data[:] = 1
    return null


def clustering_vs_random(
    G: sparse.csr_matrix, rng: np.random.Generator, n_null: int = 3
) -> tuple[float, float, float]:
    """`(observed, null_mean, ratio)` clustering (spec §5.1: "≫ random graph
    of same degree"). Averaged over `n_null` degree-preserving rewirings.
    """
    observed = global_clustering(G)
    null_values = [global_clustering(configuration_null(G, rng)) for _ in range(n_null)]
    null_mean = float(np.mean(null_values))
    ratio = float(observed / null_mean) if null_mean > 0 else float("inf")
    return observed, null_mean, ratio
