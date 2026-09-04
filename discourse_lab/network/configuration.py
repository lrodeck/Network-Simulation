"""Configuration model (spec §2.2): degree sequence only, no homophily. The
null model — structure comes purely from the heavy-tailed degree sequence, so
comparing against `latent_space` isolates what homophily itself contributes.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from discourse_lab.config import Config
from discourse_lab.population import Population
from discourse_lab.registry import register


@register("graph_generator", "configuration_model")
def configuration_model_graph(cfg: Config, pop: Population, rng: np.random.Generator) -> sparse.csr_matrix:
    n = pop.X_used.shape[0]
    prominence = pop.X_used[:, pop.trait_names.index("prominence")]

    # Followee endpoint drawn proportional to prominence (in-degree heavy tail,
    # Chung-Lu style); follower endpoint (out-degree) is degree-sequence only.
    weights = prominence / prominence.sum()
    out_deg = rng.poisson(cfg.graph.mean_degree, size=n).clip(min=0)
    m = int(out_deg.sum())
    if m == 0:
        return sparse.csr_matrix((n, n), dtype=np.int8)

    src = np.repeat(np.arange(n), out_deg)
    dst = rng.choice(n, size=m, p=weights, replace=True)

    keep = src != dst
    data = np.ones(keep.sum(), dtype=np.int8)
    G = sparse.coo_matrix((data, (src[keep], dst[keep])), shape=(n, n)).tocsr()
    G.data[:] = 1
    return G
