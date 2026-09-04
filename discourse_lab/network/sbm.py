"""Stochastic block model (spec §2.2): explicit blocks, for when you want to
*impose* community structure rather than let it emerge from trait geometry.
Blocks default to archetype membership.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from discourse_lab.config import Config
from discourse_lab.population import Population
from discourse_lab.registry import register


@register("graph_generator", "sbm")
def sbm_graph(cfg: Config, pop: Population, rng: np.random.Generator) -> sparse.csr_matrix:
    n = pop.X_used.shape[0]
    gcfg = cfg.graph

    if gcfg.sbm_blocks and gcfg.sbm_blocks > 0:
        blocks = rng.integers(0, gcfg.sbm_blocks, size=n)
        n_blocks = gcfg.sbm_blocks
    else:
        names = sorted(set(pop.archetype_names))
        idx = {name: i for i, name in enumerate(names)}
        blocks = np.array([idx[pop.archetype_names[c]] for c in pop.archetype_labels])
        n_blocks = len(names)

    # within/between edge probability calibrated so the realised mean degree
    # matches the target, with sbm_homophily setting the within:between ratio.
    counts = np.bincount(blocks, minlength=n_blocks)
    same_pairs = (counts * (counts - 1)).sum()
    diff_pairs = n * (n - 1) - same_pairs
    h = gcfg.sbm_homophily
    target_edges = gcfg.mean_degree * n

    # p_within = h * p_between; solve p_between from the total-edge constraint.
    denom = h * same_pairs + diff_pairs
    p_between = target_edges / denom if denom > 0 else 0.0
    p_within = min(h * p_between, 1.0)
    p_between = min(p_between, 1.0)

    block_of = blocks
    same_block = block_of[:, None] == block_of[None, :] if n <= 4000 else None

    if same_block is not None:
        probs = np.where(same_block, p_within, p_between)
        np.fill_diagonal(probs, 0.0)
        draws = rng.random((n, n)) < probs
        rows, cols = np.nonzero(draws)
    else:
        # large N: sample same-block and cross-block edges separately by block
        rows_list, cols_list = [], []
        order = np.argsort(blocks)
        block_starts = np.searchsorted(blocks[order], np.arange(n_blocks + 1))
        for b in range(n_blocks):
            members = order[block_starts[b] : block_starts[b + 1]]
            m = len(members)
            if m > 1:
                draws = rng.random((m, m)) < p_within
                np.fill_diagonal(draws, False)
                r, c = np.nonzero(draws)
                rows_list.append(members[r])
                cols_list.append(members[c])
        # cross-block edges via global Bernoulli thinning (approximate, cheap)
        m_cross = int(p_between * diff_pairs)
        if m_cross > 0:
            src = rng.integers(0, n, m_cross)
            dst = rng.integers(0, n, m_cross)
            keep = block_of[src] != block_of[dst]
            rows_list.append(src[keep])
            cols_list.append(dst[keep])
        rows = np.concatenate(rows_list) if rows_list else np.array([], dtype=int)
        cols = np.concatenate(cols_list) if cols_list else np.array([], dtype=int)

    data = np.ones(len(rows), dtype=np.int8)
    G = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    G.data[:] = 1
    return G
