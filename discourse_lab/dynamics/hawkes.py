"""Hawkes replies (spec §2.3, §3): reply intensity to a post is self-exciting,
not Poisson, which is what produces realistic thread burstiness — a post
gets its comments in a clump, not spread uniformly.

    lambda_p(t) = mu_p + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

`alpha/beta` (`hawkes_ratio`) must stay < 1 for stability; as it approaches 1
you get pile-on dynamics. Implemented with the standard exponential-kernel
recursion so per-tick cost is O(active threads), not O(events).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HawkesThreads:
    """One entry per open thread (root post). `excitation` is the decayed sum
    of past events (the recursive Hawkes state); `age` is ticks since the
    root post, used to close threads past `max_thread_age`.
    """

    post_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    mu: np.ndarray = field(default_factory=lambda: np.empty(0))
    excitation: np.ndarray = field(default_factory=lambda: np.empty(0))
    age: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def open_threads(
        self, post_ids: np.ndarray, mu: np.ndarray | float, excitation: np.ndarray | float = 0.0
    ) -> None:
        """Open many threads at once — one concatenation, not one per post.

        Opening them one at a time made thread bookkeeping O(n^2): every
        `np.append` reallocates and copies the whole array, and a busy tick
        opens thousands of threads. spec §3.2 rules out per-user Python loops
        in the tick for exactly this reason.
        """
        post_ids = np.atleast_1d(np.asarray(post_ids, dtype=np.int64))
        if len(post_ids) == 0:
            return
        mu_arr = np.broadcast_to(np.asarray(mu, dtype=float), post_ids.shape)
        self.post_id = np.concatenate([self.post_id, post_ids])
        self.mu = np.concatenate([self.mu, mu_arr])
        exc = np.broadcast_to(np.asarray(excitation, dtype=float), post_ids.shape)
        self.excitation = np.concatenate([self.excitation, exc])
        self.age = np.concatenate([self.age, np.zeros(len(post_ids), dtype=np.int64)])

    def open_thread(self, post_id: int, mu: float) -> None:
        self.open_threads(np.array([post_id], dtype=np.int64), mu)

    def step(self, rng: np.random.Generator, alpha: float, beta: float, max_age: int, dt: float = 1.0) -> dict[int, int]:
        """Advance one tick: decay, draw replies, excite, age out. Returns
        {post_id: n_replies} for threads that got at least one reply.
        """
        if len(self.post_id) == 0:
            return {}

        self.excitation *= np.exp(-beta * dt)
        intensity = self.mu + alpha * self.excitation
        n_replies = rng.poisson(np.clip(intensity * dt, 0, None))
        self.excitation += n_replies.astype(float)
        self.age += 1

        keep = self.age < max_age
        result = {int(pid): int(n) for pid, n in zip(self.post_id, n_replies) if n > 0}

        self.post_id, self.mu = self.post_id[keep], self.mu[keep]
        self.excitation, self.age = self.excitation[keep], self.age[keep]
        return result

    def excitation_of(self, post_ids: np.ndarray) -> np.ndarray:
        """Current *excitation* (the decaying part) for each id, 0 if closed.

        Inheritance must be seeded here and never into `mu`: `mu` is the
        permanent baseline and does not decay, so folding a parent's heat into
        a child's `mu` gives every reply a permanently raised floor and the
        process runs away. Measured with mu-inheritance at hawkes_ratio=0.6
        and n_users=800, replies/tick over ticks 0-20 / 40-60 / 100-120:

            inherit 0.00    1.1     1.1     0.8      stable
            inherit 0.10    1.9     2.2     3.4      creeping
            inherit 0.13    1.9    10.0   400.7      saturated

        Excitation decays at `beta`, so an inherited share fades like any
        other event and stability stays governed by alpha/beta < 1.
        """
        return self._lookup(post_ids, self.excitation)

    def intensity_of(self, post_ids: np.ndarray, alpha: float) -> np.ndarray:
        """Current reply intensity for each id, 0.0 where the thread is closed.

        Vectorised over ids: a linear scan per id was the other half of the
        O(n^2) — replies are looked up in bulk once per tick.
        """
        return self._lookup(post_ids, self.mu) + alpha * self._lookup(post_ids, self.excitation)

    def _lookup(self, post_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        post_ids = np.atleast_1d(np.asarray(post_ids, dtype=np.int64))
        out = np.zeros(len(post_ids))
        if len(self.post_id) == 0:
            return out
        order = np.argsort(self.post_id)
        sorted_ids = self.post_id[order]
        pos = np.clip(np.searchsorted(sorted_ids, post_ids), 0, len(sorted_ids) - 1)
        hit = sorted_ids[pos] == post_ids
        out[hit] = values[order[pos[hit]]]
        return out

    def __len__(self) -> int:
        return len(self.post_id)


def generate_reply_posts(
    targets: dict[int, int],
    active_posts,
    pop,
    expression,
    s_t: np.ndarray,
    sigma_t: np.ndarray,
    rng: np.random.Generator,
    start_id: int,
    t: int,
    max_depth: int,
) -> tuple[object | None, list[str]]:
    """Turn a Hawkes draw `{post_id: n_replies}` into a `PostBatch` of replies.

    This is spec §3.1 step 2's `replies = hawkes_draw(open_threads, t)`: reply
    posts are *generated*, scheduled by thread intensity, not derived from
    whatever the exposure pass happened to surface. The `reply` action in the
    engagement kernel (§2.6) is a separate thing — an engagement event that
    feeds the discourse-state update and drift channel 2 — and §2.7 keeps
    cascades to repost/quote alone.

    Repliers are sampled in proportion to `reply_prop`, the §1.1 behaviour
    trait that exists for exactly this and was otherwise only shaping the
    lurker archetype's offsets.
    """
    from discourse_lab.dynamics.posts import PostBatch, filter_post_batch

    warnings_out: list[str] = []
    if not targets or active_posts is None or len(active_posts) == 0:
        return None, warnings_out

    id_to_idx = {int(pid): i for i, pid in enumerate(active_posts.id)}
    parent_idx, counts = [], []
    for pid, n in targets.items():
        idx = id_to_idx.get(int(pid))
        if idx is not None:
            parent_idx.append(idx)
            counts.append(n)
    if not parent_idx:
        return None, warnings_out

    parent_idx = np.repeat(np.asarray(parent_idx, dtype=np.int64), counts)
    depth = active_posts.depth[parent_idx] + 1

    within = depth <= max_depth
    if not within.all():
        warnings_out.append(
            f"hawkes: dropped {int((~within).sum())} reply post(s) past max_cascade_depth={max_depth}"
        )
    parent_idx, depth = parent_idx[within], depth[within]
    m = len(parent_idx)
    if m == 0:
        return None, warnings_out

    names = pop.trait_names
    reply_prop = pop.X_used[:, names.index("reply_prop")]
    weights = np.clip(reply_prop, 1e-9, None)
    author = rng.choice(len(reply_prop), size=m, p=weights / weights.sum())

    topic_p = active_posts.topic[parent_idx]
    stance_cols = [i for i, n in enumerate(names) if n.startswith("stance_")]
    stance_new = pop.X_used[author][:, stance_cols]   # a reply is the replier's own stance

    dims = expression.generate(pop.X_stored[author], topic_p, s_t, rng)

    return PostBatch(
        author=author,
        topic=topic_p,
        stance=stance_new,
        **{k: dims[k] for k in
           ("arousal", "valence", "provocativeness", "novelty", "specificity", "quality", "length")},
        id=np.arange(start_id, start_id + m),
        t=np.full(m, t),
        parent=active_posts.id[parent_idx],
        root=active_posts.root[parent_idx],
        depth=depth,
        kind=np.full(m, "reply"),
        engagement_count=np.zeros(m, dtype=np.int64),
    ), warnings_out
