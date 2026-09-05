"""Perception: `F_local`, `F_global`, and the `w_u` blend (dev §6 step 7).

Not pinned down by an explicit formula in the spec (unlike population/graph/
timing/exposure) — the spec names the module (`perception.py`) and the
measures it feeds (dev §7.1: "the salience/stance agreement pair") but leaves
the blend itself to be designed. This implementation:

`F_global` is the actual discourse state `(s(t), sigma(t))` from spec §1.4 /
§2.8 — the same for every user.

`F_local(u)` is what a user's *own* feed actually showed them this tick: the
topic distribution and per-topic mean stance among the posts they were
exposed to (spec §2.5's `seen` pairs), falling back to the global value for
any topic they had no exposures on.

`w_u`, the blend weight, is the share of `u`'s exposures this tick that came
from accounts they actually follow (`tie_strength`) rather than algorithmic
injection — a user whose feed is pure "recommended for you" perceives mostly
the global conversation; a user whose feed is pure subscription perceives
mostly their own corner of it. `w_u = 0` (no exposures, or none from
followees) collapses `F_local` to `F_global` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.exposure.attention import Exposures


@dataclass
class PerceivedState:
    s_local: np.ndarray       # (N, K) each user's locally-perceived topic distribution
    sigma_local: np.ndarray   # (N, K, D) each user's locally-perceived per-topic stance
    sigma_var_local: np.ndarray  # (N, K) variance of stance *across the posts u actually saw*
    w: np.ndarray             # (N,) blend weight actually used
    s_perceived: np.ndarray   # (N, K) = w*s_local + (1-w)*s_global
    sigma_perceived: np.ndarray  # (N, K, D)


def compute_perception(
    n_users: int,
    exposures: Exposures,
    is_follower: np.ndarray,
    posts: PostBatch,
    s_global: np.ndarray,
    sigma_global: np.ndarray,
) -> PerceivedState:
    K = s_global.shape[0]
    D = sigma_global.shape[1]

    s_local = np.zeros((n_users, K))
    sigma_sum = np.zeros((n_users, K, D))
    sigma_sqsum = np.zeros((n_users, K, D))
    sigma_seen = np.zeros((n_users, K))
    w_num = np.zeros(n_users)
    w_den = np.zeros(n_users)

    if len(exposures) > 0:
        u = exposures.user_id
        topic_p = posts.topic[exposures.post_idx]
        stance_p = posts.stance[exposures.post_idx]

        np.add.at(s_local, (u, topic_p), 1.0)
        np.add.at(sigma_sum, (u, topic_p), stance_p)
        np.add.at(sigma_sqsum, (u, topic_p), stance_p**2)
        np.add.at(sigma_seen, (u, topic_p), 1.0)
        np.add.at(w_num, u, is_follower.astype(float))
        np.add.at(w_den, u, 1.0)

    row_sums = s_local.sum(axis=1, keepdims=True)
    s_local = np.divide(s_local, row_sums, out=np.zeros_like(s_local), where=row_sums > 0)

    sigma_local = np.divide(
        sigma_sum, sigma_seen[:, :, None], out=np.tile(sigma_global, (n_users, 1, 1)), where=sigma_seen[:, :, None] > 0
    )
    mean_sq = np.divide(sigma_sqsum, sigma_seen[:, :, None], out=np.zeros_like(sigma_sqsum), where=sigma_seen[:, :, None] > 0)
    # per-(user, topic) variance averaged over stance dims; 0 where a topic was seen only once or never
    sigma_var_local = np.clip((mean_sq - sigma_local**2), 0, None).mean(axis=2)

    w = np.divide(w_num, w_den, out=np.zeros(n_users), where=w_den > 0)

    s_perceived = w[:, None] * s_local + (1 - w[:, None]) * s_global[None, :]
    sigma_perceived = w[:, None, None] * sigma_local + (1 - w[:, None, None]) * sigma_global[None, :, :]

    return PerceivedState(
        s_local=s_local,
        sigma_local=sigma_local,
        sigma_var_local=sigma_var_local,
        w=w,
        s_perceived=s_perceived,
        sigma_perceived=sigma_perceived,
    )
