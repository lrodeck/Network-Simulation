"""Reciprocity pass (spec §2.2): after generation, add the reverse edge for
each edge with probability `r`. Empirical r ~ 0.2-0.4 on follow graphs.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def add_reciprocity(G: sparse.csr_matrix, r: float, rng: np.random.Generator) -> sparse.csr_matrix:
    if r <= 0:
        return G
    coo = G.tocoo()
    keep = rng.random(len(coo.row)) < r
    rev_rows, rev_cols = coo.col[keep], coo.row[keep]

    rows = np.concatenate([coo.row, rev_rows])
    cols = np.concatenate([coo.col, rev_cols])
    data = np.ones(len(rows), dtype=np.int8)

    out = sparse.coo_matrix((data, (rows, cols)), shape=G.shape).tocsr()
    out.setdiag(0)
    out.eliminate_zeros()
    out.data[:] = 1
    return out
