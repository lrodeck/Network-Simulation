"""Candidate inbox (spec §2.5a): for each active post, candidates are the
author's followers plus algorithmic injection.

    C_p = followers(author_p) union inject(p, k_inj)

`inject` samples non-followers — this is "recommended for you" and is what
makes cross-cluster spread possible. `k_inj = 0` reduces the system to pure
subscription.

Vectorised over the batch of posts (one graph column-slice per unique author,
never a per-user Python loop over the population).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.network import Graph


@dataclass
class CandidatePairs:
    post_idx: np.ndarray     # index into the PostBatch
    user_id: np.ndarray
    is_follower: np.ndarray  # False for injected (non-follower) candidates

    def __len__(self) -> int:
        return len(self.post_idx)


def candidate_inbox(
    graph: Graph,
    posts: PostBatch,
    inject_k: int,
    fanout_cap: int,
    rng: np.random.Generator,
) -> CandidatePairs:
    """Scatter each post to its author's followers, plus `inject_k` non-followers.

    Vectorised as the ragged gather spec §3.2 prescribes. The previous version
    was that section's "honest but slow" sketch: a Python loop over posts with
    a `csc.getcol(author)` per author, which profiling showed as ~2000 scipy
    sparse-submatrix constructions per tick and the single largest cost in the
    loop. Only posts whose author exceeds `fanout_cap` still need individual
    treatment, because sampling without replacement differs per post — and at
    the default mean degree of 40 against a cap of 400 that set is usually
    empty.
    """
    n = graph.n
    P = len(posts)
    if P == 0:
        empty = np.empty(0, dtype=np.int64)
        return CandidatePairs(post_idx=empty, user_id=empty, is_follower=np.empty(0, dtype=bool))

    csc = graph.csc
    authors = posts.author.astype(np.int64)
    starts = csc.indptr[authors]
    counts = csc.indptr[authors + 1] - starts

    capped = (counts > fanout_cap) & (fanout_cap > 0)
    gather_counts = np.where(capped, 0, counts)   # over-cap posts handled below

    # ragged gather: for each post, the slice csc.indices[start : start + count]
    total = int(gather_counts.sum())
    post_idx = np.repeat(np.arange(P), gather_counts)
    within = np.arange(total) - np.repeat(np.cumsum(gather_counts) - gather_counts, gather_counts)
    follower_users = csc.indices[np.repeat(starts, gather_counts) + within]

    if capped.any():
        extra_posts, extra_users = [], []
        for i in np.flatnonzero(capped):
            block = csc.indices[starts[i] : starts[i] + counts[i]]
            picked = rng.choice(block, size=fanout_cap, replace=False)
            extra_posts.append(np.full(fanout_cap, i))
            extra_users.append(picked)
        post_idx = np.concatenate([post_idx] + extra_posts)
        follower_users = np.concatenate([follower_users] + extra_users)

    if inject_k > 0:
        inj_posts = np.repeat(np.arange(P), inject_k)
        inj_users = rng.integers(0, n, P * inject_k)
        return CandidatePairs(
            post_idx=np.concatenate([post_idx, inj_posts]),
            user_id=np.concatenate([follower_users, inj_users]),
            is_follower=np.concatenate(
                [np.ones(len(post_idx), dtype=bool), np.zeros(len(inj_posts), dtype=bool)]
            ),
        )

    return CandidatePairs(
        post_idx=post_idx,
        user_id=follower_users,
        is_follower=np.ones(len(post_idx), dtype=bool),
    )
