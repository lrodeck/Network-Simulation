"""Discourse state update (spec §2.8).

    s(t+1)    = rho_s * s(t) + (1 - rho_s) * normalize(sum_p w_p * topic_p)
    sigma(t+1)[k] = rho_sigma * sigma(t)[k] + (1 - rho_sigma) * weighted_mean(stance_p : topic_p = k)
    w_p       = engagement_count_p

Weighting by engagement rather than post count is what lets a small number
of highly-engaged users capture the agenda.
"""

from __future__ import annotations

import numpy as np

from discourse_lab.dynamics.posts import PostBatch


def update_discourse(
    s: np.ndarray,
    sigma: np.ndarray,
    posts: PostBatch | None,
    rho_s: float,
    rho_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """`posts` is None on a tick that produced no engagement. The decay still
    runs — spec §3.1 puts `decay_discourse` at step 1, unconditionally, so a
    quiet tick lets attention fade rather than freezing the agenda."""
    K = s.shape[0]
    if posts is None:
        return rho_s * s, sigma.copy()

    w = posts.engagement_count.astype(float) if len(posts) > 0 else np.zeros(0)

    if len(posts) == 0 or w.sum() == 0:
        topic_dist = np.zeros(K)
    else:
        topic_energy = np.bincount(posts.topic, weights=w, minlength=K)
        topic_dist = topic_energy / topic_energy.sum()

    s_new = rho_s * s + (1 - rho_s) * topic_dist

    sigma_new = sigma.copy()
    if len(posts) > 0:
        for k in range(K):
            mask = (posts.topic == k) & (w > 0)
            if mask.any():
                weighted_mean = np.average(posts.stance[mask], axis=0, weights=w[mask])
                sigma_new[k] = rho_sigma * sigma[k] + (1 - rho_sigma) * weighted_mean

    return s_new, sigma_new
