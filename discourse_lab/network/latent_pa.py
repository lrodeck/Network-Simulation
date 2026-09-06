"""Latent space + explicit preferential attachment (spec §0.2, §2.2).

A fifth registered generator, not a change to `latent_space`. §2.2 specifies
that generator's long ties as *uniform*, and that is left alone; this one
exists because uniform long ties cap the in-degree tail and spec §5.1's
engagement rows cannot be reached underneath that cap.

What it does NOT solve, recorded so the mistake is not repeated. This was
built on the hypothesis that spec §5.1's engagement alpha and attention Gini
were unreachable because audiences were not heavy-tailed. That hypothesis is
wrong. Raising max in-degree from 211 to 2132 moved engagement alpha only
5.66 -> 4.65 and left attention Gini flat at 0.601 -> 0.609, while costing
clustering (5.42 -> 1.88) and reciprocity. Lifting `fanout_cap` to 5000 on
top changed nothing either. Attention concentration is produced by the FEED,
not the graph: under the default chronological ranker the maximum engagement
any post ever received was 15, because a time-ordered feed spreads exposure
evenly across active posts regardless of who wrote them. Switching to
`engagement_optimized` + `bandwagon` takes that maximum to 733 and the Gini
to 0.887, in range. Use this generator for a heavy-tailed degree
distribution, which is a real thing to want; do not use it expecting §5.1's
engagement rows to move.

The problem it addresses. `prominence` enters as Pareto(2.30) with a max/mean of
303x, but `latent_space` yields in-degree at alpha ~5-8 with a max only ~4x
the mean: inside the kNN pool a user can only be followed by the ~knn_k users
whose latent neighbourhood contains them, so the `gamma * log(1 + prominence)`
term reorders candidates without ever lifting anyone out of that geometric
ceiling.

The mechanism. Local edges come from the same kNN draw, preserving the
clustering and homophily §5.1 also asks for. A `pa_fraction` share of edges is
then drawn globally with destination probability proportional to prominence.
Sampling m destinations with p_i ∝ w_i gives E[deg_i] ∝ w_i, so when w is
Pareto(alpha) the in-degree tail inherits that alpha — which is why this
reaches the target range and a uniform draw cannot.

`pa_fraction` is the dial between the two regimes: 0 is (almost) plain kNN,
1 is (almost) pure preferential attachment with no local structure. The
clustering ratio falls as it rises, so it trades §5.1's clustering row against
its engagement rows.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import expit

from discourse_lab.config import Config
from discourse_lab.network.latent_space import _calibrate_alpha, _candidate_edges
from discourse_lab.population import Population
from discourse_lab.registry import register


@register("graph_generator", "latent_pa")
def latent_pa_graph(cfg: Config, pop: Population, rng: np.random.Generator) -> sparse.csr_matrix:
    gcfg = cfg.graph
    n = pop.X_used.shape[0]
    prominence = pop.X_used[:, pop.trait_names.index("prominence")]

    pa_fraction = float(np.clip(gcfg.pa_fraction, 0.0, 1.0))
    local_target = gcfg.mean_degree * (1.0 - pa_fraction)

    rows = np.empty(0, dtype=np.int64)
    cols = np.empty(0, dtype=np.int64)

    if local_target > 0:
        src, dst, dist = _candidate_edges(pop, gcfg.knn_k)
        logit_base = -gcfg.homophily_beta * dist + gcfg.prominence_gamma * np.log1p(prominence[dst])
        alpha = _calibrate_alpha(logit_base, local_target, n, rng)
        draws = rng.random(logit_base.shape) < expit(logit_base + alpha)
        rows, cols = src[draws], dst[draws]

    n_pa = int(round(gcfg.mean_degree * pa_fraction * n))
    if n_pa > 0:
        weights = prominence / prominence.sum()
        pa_rows = rng.integers(0, n, n_pa)
        pa_cols = rng.choice(n, size=n_pa, p=weights)
        keep = pa_rows != pa_cols
        rows = np.concatenate([rows, pa_rows[keep]])
        cols = np.concatenate([cols, pa_cols[keep]])

    G = sparse.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)).tocsr()
    G.setdiag(0)
    G.eliminate_zeros()
    G.data[:] = 1
    return G
