"""Tie rewiring (change spec C2.2): follow/unfollow as a slow, periodic,
drift-adjacent process.

    P(unfollow) ~ accumulated hostile interaction with that author
    P(follow)   ~ accumulated positive engagement with a non-followed author

Static graphs are the dev doc's settled v1 choice, and dynamic rewiring is
the change spec's C2: self-sorting into echo chambers through network
rewriting, the mechanism Törnberg (PNAS 2022) locates behind affective
polarization. It runs every `rewire_every` ticks, not every tick — it is a
slow process, and a per-tick sparse-matrix rebuild at N=1e4 is the one thing
here that could plausibly dominate runtime.

Cache discipline: the *initial* graph stays a content-addressed artifact
(io/artifacts.py). Rewiring mutates the run's own Graph object — freshly
loaded from disk for every run, so no artifact is ever touched — and the
final state plus an edge-change log are persisted in the run directory.
Every clustering/reciprocity metric computed on a rewired run must say which
snapshot it read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from discourse_lab.network import Graph

# actions that count as hostile contact with the author, and as positive
# contact; skips are not interactions and count for neither.
HOSTILE_ACTIONS = ("reply", "quote", "report")
POSITIVE_ACTIONS = ("like", "repost")


@dataclass
class RewireState:
    """Accumulators over (user, author) pairs since the last rewire. Kept as
    appended per-tick arrays and concatenated at rewire time — a dense N×N
    accumulator would be 800 MB at N=1e4 for a matrix that is ~99.9% empty."""

    user: list[np.ndarray] = field(default_factory=list)
    author: list[np.ndarray] = field(default_factory=list)
    valence: list[np.ndarray] = field(default_factory=list)
    # (t, user, target, added) rows for the edge-change log
    events: list[tuple[int, int, int, bool]] = field(default_factory=list)

    def observe(self, users: np.ndarray, post_author: np.ndarray, actions: np.ndarray) -> None:
        """Accumulate this tick's engagement valence per (user, author)."""
        if len(users) == 0:
            return
        valence = np.where(np.isin(actions, HOSTILE_ACTIONS), -1.0, 0.0)
        valence = np.where(np.isin(actions, POSITIVE_ACTIONS), 1.0, valence)
        keep = valence != 0.0
        if not keep.any():
            return
        self.user.append(users[keep].astype(np.int64))
        self.author.append(post_author[keep].astype(np.int64))
        self.valence.append(valence[keep])

    def maybe_rewire(self, t: int, graph: Graph, cfg, rng: np.random.Generator) -> bool:
        """Rewire on the cadence `rewire_every` sets. Returns True if it ran."""
        dcfg = cfg.dynamics
        if not dcfg.rewire or dcfg.rewire_every <= 0 or t == 0:
            return False
        if t % dcfg.rewire_every != 0:
            return False
        self.rewire(t, graph, dcfg.rewire_rate, rng)
        return True

    def rewire(self, t: int, graph: Graph, rate: float, rng: np.random.Generator) -> None:
        if not self.user:
            return
        users = np.concatenate(self.user)
        authors = np.concatenate(self.author)
        valence = np.concatenate(self.valence)
        self.user, self.author, self.valence = [], [], []

        n = graph.n
        # accumulate signed valence per (user, author) pair
        key = users.astype(np.int64) * n + authors.astype(np.int64)
        uniq, inv = np.unique(key, return_inverse=True)
        val = np.zeros(len(uniq))
        np.add.at(val, inv, valence)
        pu = (uniq // n).astype(np.int64)
        pa = (uniq % n).astype(np.int64)

        # --- unfollow: hostile accumulation on an existing edge -----------
        # membership of each UNIQUE (u, a) pair in the edge set, vectorised
        # via edge keys
        rows_all, cols_all = graph.csr.nonzero()
        edge_key_set = np.sort(rows_all.astype(np.int64) * n + cols_all.astype(np.int64))
        pos = np.searchsorted(edge_key_set, uniq)
        pos = np.clip(pos, 0, len(edge_key_set) - 1)
        is_edge = edge_key_set[pos] == uniq

        hostile_edge = is_edge & (val < 0)
        drop = np.zeros(len(uniq), dtype=bool)
        if hostile_edge.any():
            p_drop = np.clip(rate * (-val[hostile_edge]), 0.0, 1.0)
            drop_idx_local = np.flatnonzero(hostile_edge)[rng.random(int(hostile_edge.sum())) < p_drop]
            drop[drop_idx_local] = True

        # --- follow: positive accumulation on a non-edge -------------------
        positive_nonedge = (~is_edge) & (val > 0)
        add = np.zeros(len(uniq), dtype=bool)
        if positive_nonedge.any():
            p_add = np.clip(rate * val[positive_nonedge], 0.0, 1.0)
            add_idx_local = np.flatnonzero(positive_nonedge)[rng.random(int(positive_nonedge.sum())) < p_add]
            add[add_idx_local] = True

        if not (drop.any() or add.any()):
            return

        kept_keys = set(edge_key_set.tolist())
        for k in uniq[drop]:
            kept_keys.discard(int(k))
        for k in uniq[add]:
            kept_keys.add(int(k))
        for k in uniq[drop]:
            self.events.append((t, int(k // n), int(k % n), False))
        for k in uniq[add]:
            self.events.append((t, int(k // n), int(k % n), True))

        new_rows = (np.fromiter(kept_keys, dtype=np.int64) // n)
        new_cols = (np.fromiter(kept_keys, dtype=np.int64) % n)
        G = sparse.coo_matrix(
            (np.ones(len(new_rows), dtype=np.int8), (new_rows, new_cols)), shape=(n, n)
        ).tocsr()
        G.setdiag(0)
        G.eliminate_zeros()
        G.data[:] = 1
        graph.csr = G
        graph.csc = G.tocsc()
