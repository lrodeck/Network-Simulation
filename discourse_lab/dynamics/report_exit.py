"""`report` as an exit event (change spec V4), not a hostility increment.

V1's warrant: reporting is *dis*engagement — the user hands the conflict to
the platform and exits — so modelling it as the largest animus increment (the
pre-V2 table's `report: 2.0`) looks like reverse causation compiled forward:
one reports *because* angry, not the other way around. V2 removed `report`
from the hostility table entirely. This module supplies what replaces it:

    propensity   report utility gains an `outgroup_x_animus` term (wired
                 into the `outrage` kernel in exposure/kernel.py) so
                 reporting rises with the reporter's own animus — the
                 correct causal direction.
    effect       the reporting user's future exposure to the reported
                 AUTHOR drops (`ReportSuppressionState`), gated on
                 `dynamics.report_exit`. The spec also allows suppressing by
                 content class instead of by author; author-level is
                 implemented here as the simpler, primary reading — a
                 topic/content-class version is not.

`report_animus_increment` (a free parameter, defaulting to 0 per the spec) is
applied directly in `dynamics/drift.py::affect_delta`, bypassing the
(action, valence) hostility table entirely, since V2's whole point was to
take `report` out of that table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from discourse_lab.exposure.inbox import CandidatePairs


@dataclass
class ReportSuppressionState:
    """(user, author) pairs the user has ever reported, as a sorted key set
    (`user * n + author`) — the same trick `dynamics/rewire.py::RewireState`
    uses to avoid an N x N dense accumulator that is ~99.9% empty, and to
    keep membership tests vectorised (`np.searchsorted`) rather than a
    per-pair Python loop.
    """

    n: int
    _keys: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def observe(self, users: np.ndarray, authors: np.ndarray) -> None:
        if len(users) == 0:
            return
        new_keys = users.astype(np.int64) * self.n + authors.astype(np.int64)
        self._keys = np.unique(np.concatenate([self._keys, new_keys]))

    def is_blocked(self, users: np.ndarray, authors: np.ndarray) -> np.ndarray:
        if len(self._keys) == 0 or len(users) == 0:
            return np.zeros(len(users), dtype=bool)
        keys = users.astype(np.int64) * self.n + authors.astype(np.int64)
        pos = np.clip(np.searchsorted(self._keys, keys), 0, len(self._keys) - 1)
        return self._keys[pos] == keys

    def filter_candidates(self, pairs: CandidatePairs, author_of_post: np.ndarray) -> CandidatePairs:
        """Drop candidate (user, post) pairs whose (user, author) was ever
        reported — the mechanism's effect: lower subsequent exposure to the
        reported source, not a lower engagement probability on it."""
        if len(pairs) == 0 or len(self._keys) == 0:
            return pairs
        blocked = self.is_blocked(pairs.user_id, author_of_post[pairs.post_idx])
        if not blocked.any():
            return pairs
        keep = ~blocked
        return CandidatePairs(
            post_idx=pairs.post_idx[keep], user_id=pairs.user_id[keep],
            is_follower=pairs.is_follower[keep],
        )
