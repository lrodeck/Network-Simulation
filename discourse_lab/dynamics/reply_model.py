"""Reply contagion models (change spec C8): simple vs complex contagion.

The Hawkes process is a self-exciting SIMPLE contagion: intensity depends on
accumulated events, not on distinct sources. Centola & Macy (2007) and
Centola (2010) show behaviours often require multiple reinforcing exposures
from DISTINCT neighbours, and — crucially — that long/weak ties slow complex
contagion, inverting Granovetter. Shipping only `hawkes` hard-codes an
answer to a live empirical question and makes every prediction about how
clustering and `inject_k` affect spread conditional on an invisible choice.

Two registered models:

    hawkes     current behaviour (spec §2.3); the default
    threshold  reply propensity rises super-linearly in the count of
               DISTINCT already-engaged in-neighbours, not in total thread
               intensity. The separating experiment (change spec D5) is the
               crossover: hawkes spreads faster across long ties, threshold
               faster inside high clustering. If both respond identically to
               clustering, `threshold` is not actually complex — check that
               the neighbour count is distinct.

The distinguishing state is a per-(user, thread) count of distinct engaged
in-neighbours — a bounded sparse structure at N=1e4 with `max_thread_age=15`,
accumulated from the engagement events the tick already produces. Never a
per-user loop.

Calibration note (change spec C8): the existing stylized-fact calibration
(>90% singletons, depth 1.5-3) was fitted with the Hawkes model; `threshold`
needs its own calibration to the same stylized facts before any D5 result
is reported. Its response scale is `THRESHOLD_SCALE` below and is a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from discourse_lab.registry import register

# count^2 / THRESHOLD_SCALE, clipped — super-linear in distinct neighbours
# (Centola's complex contagion), saturating so a celebrity thread does not
# auto-infect the whole follower graph.
THRESHOLD_SCALE = 16.0
THRESHOLD_P_MAX = 0.5


@dataclass
class ThresholdState:
    """Distinct engagers per post, accumulated from the engagement event log
    and pruned on thread age. Stored as appended (post_id, user, t) arrays —
    the engagement log is ~50x smaller than the exposure log (spec §3.5), so
    this stays small."""

    post: list[np.ndarray] = field(default_factory=list)
    user: list[np.ndarray] = field(default_factory=list)
    t: list[np.ndarray] = field(default_factory=list)

    def observe(self, t: int, users: np.ndarray, post_ids: np.ndarray, actions: np.ndarray) -> None:
        keep = actions != "skip"
        if not keep.any():
            return
        self.post.append(np.asarray(post_ids[keep], dtype=np.int64))
        self.user.append(np.asarray(users[keep], dtype=np.int64))
        self.t.append(np.full(int(keep.sum()), t, dtype=np.int64))

    def prune(self, t_now: int, max_age: int) -> None:
        if not self.t:
            return
        all_t = np.concatenate(self.t)
        keep = (t_now - all_t) <= max_age
        if keep.all():
            return
        flat_post = np.concatenate(self.post)[keep]
        flat_user = np.concatenate(self.user)[keep]
        flat_t = all_t[keep]
        self.post = [flat_post]
        self.user = [flat_user]
        self.t = [flat_t]


def hawkes_draw(threads, rng: np.random.Generator, alpha: float, beta: float,
                max_age: int, max_replies_per_tick: int) -> dict[int, int]:
    """The existing simple-contagion draw, registered for completeness."""
    return threads.step(rng, alpha, beta, max_age, max_replies_per_tick=max_replies_per_tick)


@register("reply_model", "hawkes")
def hawkes_model(threads, rng: np.random.Generator, alpha: float, beta: float,
                 max_age: int, max_replies_per_tick: int, **_) -> tuple[dict[int, int], dict[int, np.ndarray]]:
    return hawkes_draw(threads, rng, alpha, beta, max_age, max_replies_per_tick), {}


@register("reply_model", "threshold")
def threshold_model(state: "ThresholdState", graph, active_posts, reply_prop: np.ndarray,
                    rng: np.random.Generator, max_age: int,
                    max_replies_per_tick: int, threshold_scale: float = 16.0,
                    **_) -> tuple[dict[int, int], dict[int, np.ndarray]]:
    """Complex-contagion draw. For each thread with at least one engaged
    user, candidates are the followers of engagers; a candidate's reply
    propensity rises super-linearly in how many DISTINCT engagers they
    follow. Chosen authors are returned per thread, so the reply posts are
    written by the users the contagion actually reached — under `hawkes`,
    authors are a reply_prop-weighted population sample instead.

    `threshold_scale` is `cfg.dynamics.threshold_scale`: the response curve
    is `reply_prop * count^2 / scale`, and the scale needs its own
    calibration to the §5.1 stylized facts before any D5 crossover result is
    reported — a config field so that calibration is a sweep, not a code
    edit."""
    targets: dict[int, int] = {}
    authors: dict[int, np.ndarray] = {}
    if not state.post or active_posts is None or len(active_posts) == 0:
        return targets, authors

    flat_post = np.concatenate(state.post)
    flat_user = np.concatenate(state.user)

    id_to_idx = {int(pid): i for i, pid in enumerate(active_posts.id)}
    # engagers grouped per post (threads with >=1 engager only)
    order = np.argsort(flat_post, kind="stable")
    sorted_posts = flat_post[order]
    starts = np.flatnonzero(np.r_[True, sorted_posts[1:] != sorted_posts[:-1]])
    uniq_posts = sorted_posts[starts]

    csc = graph.csc
    n = graph.n
    for s0, pid in zip(starts, uniq_posts):
        seg_end = int(np.searchsorted(sorted_posts, pid, side="right"))
        engagers = np.unique(flat_user[order[s0:seg_end]])
        idx = id_to_idx.get(int(pid))
        if idx is None or len(engagers) == 0:
            continue
        # candidates: followers of each engager, tagged by engager, so the
        # DISTINCT count per candidate is a unique-(u, v) tally
        cand_parts, tag_parts = [], []
        for v in engagers:
            v = int(v)
            lo, hi = csc.indptr[v], csc.indptr[v + 1]
            if hi > lo:
                cand_parts.append(csc.indices[lo:hi])
                tag_parts.append(np.full(hi - lo, v, dtype=np.int64))
        if not cand_parts:
            continue
        cand = np.concatenate(cand_parts)
        tag = np.concatenate(tag_parts)
        pair = cand.astype(np.int64) * n + tag
        upair, inverse = np.unique(pair, return_inverse=True)
        distinct = np.zeros(len(upair))
        np.add.at(distinct, inverse, 1.0)
        cand_users = (upair // n).astype(np.int64)

        # super-linear in distinct engagers, gated by the user's own reply
        # propensity, saturating at THRESHOLD_P_MAX
        p = np.clip(
            reply_prop[cand_users] * (distinct**2) / max(threshold_scale, 1e-9),
            0.0,
            THRESHOLD_P_MAX,
        )
        draws = rng.random(len(cand_users)) < p
        chosen = cand_users[draws]
        if len(chosen) == 0:
            continue
        if max_replies_per_tick > 0:
            chosen = chosen[:max_replies_per_tick]
        targets[int(pid)] = len(chosen)
        authors[int(pid)] = chosen
    return targets, authors


def reply_model_names() -> list[str]:
    from discourse_lab.registry import names

    return names("reply_model")
