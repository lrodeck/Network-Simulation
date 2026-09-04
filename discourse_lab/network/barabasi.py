"""Barabasi-Albert preferential attachment (spec §2.2): pure preferential
attachment, no homophily. Grows an undirected backbone (each new node attaches
to m existing nodes with probability proportional to current degree), then
each undirected edge becomes one directed follow edge with a random
orientation — the reciprocity pass adds the rest.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from discourse_lab.config import Config
from discourse_lab.population import Population
from discourse_lab.registry import register


@register("graph_generator", "barabasi_albert")
def barabasi_albert_graph(cfg: Config, pop: Population, rng: np.random.Generator) -> sparse.csr_matrix:
    n = pop.X_used.shape[0]
    m = max(1, round(cfg.graph.mean_degree / 2))
    m = min(m, n - 1)

    # classic BA implementation: start with a small complete seed so early
    # nodes have attachment targets, tracking a repeated-node list whose
    # sampling frequency is proportional to current degree.
    seed_size = m + 1
    src_list: list[int] = []
    dst_list: list[int] = []
    repeated_nodes: list[int] = []
    for i in range(seed_size):
        for j in range(i + 1, seed_size):
            src_list.append(i)
            dst_list.append(j)
            repeated_nodes.append(i)
            repeated_nodes.append(j)

    for new_node in range(seed_size, n):
        chosen = rng.choice(repeated_nodes, size=m, replace=len(repeated_nodes) < m)
        chosen = np.unique(chosen)
        while len(chosen) < m:
            extra = rng.integers(0, new_node)
            if extra not in chosen:
                chosen = np.append(chosen, extra)
        for target in chosen[:m]:
            src_list.append(new_node)
            dst_list.append(int(target))
            repeated_nodes.append(new_node)
            repeated_nodes.append(int(target))

    src = np.array(src_list)
    dst = np.array(dst_list)
    flip = rng.random(len(src)) < 0.5
    rows = np.where(flip, src, dst)
    cols = np.where(flip, dst, src)

    data = np.ones(len(rows), dtype=np.int8)
    G = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    G.data[:] = 1
    return G
