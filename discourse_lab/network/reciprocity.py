"""Reciprocity pass (spec §2.2): after generation, add the reverse edge for
each edge with probability `mirror_p`.

`mirror_p` is NOT the reciprocity you then measure. Mirroring a fraction r of
E edges leaves 2rE reciprocated edges out of E(1+r) total, i.e. a measured
share of `2r / (1 + r)` — and the homophilous generator already produces
reciprocal pairs by chance on top of that, ~0.157 at the default settings
with this pass switched off entirely. Spec §5.1's 0.2-0.4 target is the
measured share, so `mirror_p = 0.10` (measuring ~0.30) is what hits it, not
`0.2`. The config field was called `reciprocity` and read as if it were the
measured quantity; it is renamed to keep the two apart.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def add_reciprocity(G: sparse.csr_matrix, mirror_p: float, rng: np.random.Generator) -> sparse.csr_matrix:
    if mirror_p <= 0:
        return G
    coo = G.tocoo()
    keep = rng.random(len(coo.row)) < mirror_p
    rev_rows, rev_cols = coo.col[keep], coo.row[keep]

    rows = np.concatenate([coo.row, rev_rows])
    cols = np.concatenate([coo.col, rev_cols])
    data = np.ones(len(rows), dtype=np.int8)

    out = sparse.coo_matrix((data, (rows, cols)), shape=G.shape).tocsr()
    out.setdiag(0)
    out.eliminate_zeros()
    out.data[:] = 1
    return out
