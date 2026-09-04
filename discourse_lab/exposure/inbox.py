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
    n = graph.n
    unique_authors = np.unique(posts.author)
    followers_by_author = {int(a): graph.csc.getcol(int(a)).indices for a in unique_authors}

    post_idx_parts, user_parts, is_follower_parts = [], [], []
    for i in range(len(posts)):
        author = int(posts.author[i])
        followers = followers_by_author[author]
        if len(followers) > fanout_cap > 0:
            followers = rng.choice(followers, size=fanout_cap, replace=False)

        injected = rng.integers(0, n, inject_k) if inject_k > 0 else np.empty(0, dtype=np.int64)

        candidates = np.concatenate([followers, injected])
        flags = np.concatenate([np.ones(len(followers), dtype=bool), np.zeros(len(injected), dtype=bool)])

        post_idx_parts.append(np.full(len(candidates), i))
        user_parts.append(candidates)
        is_follower_parts.append(flags)

    if not post_idx_parts:
        empty = np.empty(0, dtype=np.int64)
        return CandidatePairs(post_idx=empty, user_id=empty, is_follower=np.empty(0, dtype=bool))

    return CandidatePairs(
        post_idx=np.concatenate(post_idx_parts),
        user_id=np.concatenate(user_parts),
        is_follower=np.concatenate(is_follower_parts),
    )
