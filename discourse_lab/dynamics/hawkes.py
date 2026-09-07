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

    def step(
        self,
        rng: np.random.Generator,
        alpha: float,
        beta: float,
        max_age: int,
        dt: float = 1.0,
        max_replies_per_tick: int = 0,
        draw: bool = True,
        mu_weights: np.ndarray | None = None,
    ) -> dict[int, int]:
        """Advance one tick: decay, draw replies, excite, age out. Returns
        {post_id: n_replies} for threads that got at least one reply.

        `max_replies_per_tick` caps arrivals per post per tick (0 = uncapped
        Poisson). At 1 the draw becomes Bernoulli with the same per-tick
        probability, 1 - exp(-lambda), which is the discrete-time reading of
        the same point process and produces a very different *shape*:

        Uncapped, a hot post collects many simultaneous children, so a thread
        grows as a bush and volume explodes before depth does. Measured, every
        setting that raised thread depth toward spec §5.1's 1.5-3 was
        supercritical: at hawkes_mu_inherit 1.8 replies per tick went 6 -> 4453
        by tick 30, and at 2.5 the run exhausted memory. Capped, a post is
        replied to at most once per tick, so a conversation extends as a chain
        — which is what a real reply thread is, and what produces depth
        without runaway volume.

        `draw=False` (C8) runs the bookkeeping — decay and ageing — without
        drawing from the intensity: the threshold reply model replaces the
        intensity draw with its own distinct-engaged-neighbour rule but still
        needs the Hawkes state to age.

        `mu_weights` (C13a) reweights the baseline per open thread — used to
        condition the draw on live candidate availability. Without it, the
        flat-mu draw targets every open thread uniformly, and ~90% of draws
        land on threads past their exposure window whose candidate pool is
        empty: the reply then comes from the global lottery, which is exactly
        the content-blind artefact C13a exists to remove. A thread nobody is
        engaging with has no audience to reply.
        """
        if len(self.post_id) == 0:
            return {}

        self.excitation *= np.exp(-beta * dt)
        mu_eff = self.mu if mu_weights is None else self.mu * mu_weights
        intensity = mu_eff + alpha * self.excitation
        rate = np.clip(intensity * dt, 0, None)
        if not draw:
            n_replies = np.zeros(len(rate), dtype=np.int64)
        elif max_replies_per_tick == 1:
            n_replies = (rng.random(len(rate)) < -np.expm1(-rate)).astype(np.int64)
        else:
            n_replies = rng.poisson(rate)
            if max_replies_per_tick > 1:
                n_replies = np.minimum(n_replies, max_replies_per_tick)
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


def thread_sigma_local(posts, stance_cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """C13c: `sigma_local(root_id)` — engagement-weighted mean stance of the
    posts ALREADY in each thread, one row per distinct root.

    Weights are `engagement_count + 1`: every post carries base weight 1, so
    a thread with no engagement yet is its own unweighted mean rather than
    undefined, and engagement tilts the room toward what landed. The reply
    being generated is never in `posts` — it does not exist yet — so the
    blend cannot see itself.

    Returns (unique_root_ids, sigma_local matrix) for vectorised lookup.
    """
    if len(posts) == 0 or len(stance_cols) == 0:
        e = np.empty(0, dtype=np.int64)
        return e, np.zeros((0, len(stance_cols)))
    root_ids = np.where(posts.root >= 0, posts.root, posts.id).astype(np.int64)
    stance = posts.stance
    w = posts.engagement_count.astype(np.float64) + 1.0

    uniq, inv = np.unique(root_ids, return_inverse=True)
    acc = np.zeros((len(uniq), len(stance_cols)))
    wsum = np.zeros(len(uniq))
    np.add.at(acc, inv, w[:, None] * stance)
    np.add.at(wsum, inv, w)
    sigma = acc / wsum[:, None]
    return uniq, sigma


def content_mu_multiplier(
    posts,
    pool,
    gamma: tuple[tuple[str, float], ...],
) -> np.ndarray:
    """C13b: reply intensity conditioned on the post's own dimensions.

        mu_p = mu_base * exp( g_prov*prov  + g_arousal*arousal
                              + g_disagree*||stance_p - stance_root|| )

    `gamma` pairs are ("prov"|"arousal"|"disagree", coefficient); an empty
    tuple returns exact ones (flat mu, the pre-C13 behaviour). Distance runs
    to the THREAD ROOT, not the parent: what makes a reply generative in the
    DMP account is that it contests what the arena formed around. Roots sit
    at distance 0 from their own arena, so the disagree term is inert for
    them — they define it. `pool` is the active-post batch the root stance
    is looked up in; a miss (root not in the pool yet) reads as distance 0,
    which is exactly right for a root opening its own thread.

    Conditioning raises variance in mu, which moves the singleton share and
    the depth distribution: recalibrate `hawkes_ratio` / `hawkes_mu_inherit`
    after switching this on (C13 dev notes §1.5 — inherit LAST, it is where
    the reply process goes supercritical).
    """
    n = len(posts.author) if hasattr(posts, "author") else 0
    if not gamma or n == 0:
        return np.ones(max(n, 0))
    g = dict(gamma)
    log_mu = np.zeros(n)
    if "prov" in g:
        log_mu += g["prov"] * posts.provocativeness
    if "arousal" in g:
        log_mu += g["arousal"] * posts.arousal
    if "disagree" in g and posts.stance.shape[1] > 0:
        root_ids = np.where(posts.root >= 0, posts.root, posts.id).astype(np.int64)
        pool_ids = pool.id
        order = np.argsort(pool_ids)
        sorted_ids = pool_ids[order]
        pos = np.clip(np.searchsorted(sorted_ids, root_ids), 0, max(len(sorted_ids) - 1, 0))
        hit = (len(sorted_ids) > 0) & (sorted_ids[pos] == root_ids)
        d = np.zeros(n)
        if hit.any():
            root_stance_p = np.zeros((n, posts.stance.shape[1]))
            root_stance_p[hit] = pool.stance[order[pos[hit]]]
            # full-row norm, then zero the misses (a root not in the pool is
            # opening its own arena: distance 0 by definition)
            d = np.where(hit, np.linalg.norm(posts.stance - root_stance_p, axis=1), 0.0)
        log_mu += g["disagree"] * d
    return np.exp(np.clip(log_mu, -5.0, 5.0))


def _sigma_local_lookup(
    sigma_local_root: tuple[np.ndarray, np.ndarray] | None,
    root_ids: np.ndarray, D: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(rows, miss) — σ_local per reply looked up by thread root id, with a
    boolean mask for roots absent from the table (caller falls back to
    σ(t) on those rows)."""
    if not sigma_local_root or len(sigma_local_root[0]) == 0:
        return np.zeros((len(root_ids), D)), np.ones(len(root_ids), dtype=bool)
    uniq, table = sigma_local_root
    pos = np.clip(np.searchsorted(uniq, root_ids), 0, len(uniq) - 1)
    miss = uniq[pos] != root_ids
    rows = table[pos]
    return rows, miss


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
    authors_for_target: dict[int, np.ndarray] | None = None,
    reply_conformity: str = "global",
    reply_conformity_mix: float = 0.5,
    sigma_local_root: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[object | None, list[str]]:
    """Turn a reply draw `{post_id: n_replies}` into a `PostBatch` of replies.

    This is spec §3.1 step 2's `replies = hawkes_draw(open_threads, t)`: reply
    posts are *generated*, scheduled by the reply model (§2.3's Hawkes
    intensity under the default `hawkes` model), not derived from whatever
    the exposure pass happened to surface. The `reply` action in the
    engagement kernel (§2.6) is a separate thing — an engagement event that
    feeds the discourse-state update and drift channel 2 — and §2.7 keeps
    cascades to repost/quote alone.

    Repliers are sampled in proportion to `reply_prop`, the §1.1 behaviour
    trait that exists for exactly this and was otherwise only shaping the
    lurker archetype's offsets. Under the C8 `threshold` reply model the
    caller supplies `authors_for_target` instead: the contagion rule picks
    the users the distinct-engaged-neighbour count reached, and this function
    writes their replies.
    """
    from discourse_lab.dynamics.posts import PostBatch, filter_post_batch

    warnings_out: list[str] = []
    if not targets or active_posts is None or len(active_posts) == 0:
        return None, warnings_out

    id_to_idx = {int(pid): i for i, pid in enumerate(active_posts.id)}
    parent_idx, counts, override_authors = [], [], []
    for pid, n in targets.items():
        idx = id_to_idx.get(int(pid))
        if idx is not None:
            parent_idx.append(idx)
            counts.append(n)
            if authors_for_target and int(pid) in authors_for_target:
                override_authors.append(np.asarray(authors_for_target[int(pid)], dtype=np.int64))
            else:
                override_authors.append(None)
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

    # C8: under the threshold model the reply model already chose the authors
    # (the users the distinct-engaged-neighbour count reached), one array of
    # length n per target; expand to the repeated order, apply the same depth
    # filter, and fall back to reply_prop sampling for any slot it left
    # unassigned (-1).
    if any(a is not None for a in override_authors):
        rep = np.concatenate(
            [ov if ov is not None else np.full(n, -1, dtype=np.int64)
             for ov, n in zip(override_authors, counts)]
        )
        rep = rep[within]
        author = np.where(
            rep >= 0, rep,
            rng.choice(len(reply_prop), size=m, p=weights / weights.sum()),
        )
    else:
        author = rng.choice(len(reply_prop), size=m, p=weights / weights.sum())

    topic_p = active_posts.topic[parent_idx]
    stance_cols = [i for i, n in enumerate(names) if n.startswith("stance_")]

    # C13c: which room does a low-conviction replier conform to? The root
    # post's own conformity line blends toward sigma(t) and that is right for
    # it — it opened the arena. A reply is IN an arena: "local" blends toward
    # the thread's own engagement-weighted mean stance (sigma_local, computed
    # from posts already in the pool — the reply being generated is not in it
    # yet, so the blend cannot see itself), "global" toward the platform-wide
    # sigma(t), "blend" mixes the two by reply_conformity_mix. Conviction
    # weighting and noise match sample_stance exactly.
    conv = pop.X_used[author][:, names.index("conviction")][:, None]
    own_stance = pop.X_used[author][:, stance_cols]
    noise = rng.normal(0.0, 0.1, size=(m, len(stance_cols)))
    if reply_conformity == "own":
        # pre-C13 behaviour: the replier's position, no room at all
        stance_new = own_stance + noise
    else:
        local, miss = _sigma_local_lookup(sigma_local_root,
                                          active_posts.root[parent_idx],
                                          len(stance_cols))
        global_room = sigma_t[topic_p]
        if reply_conformity == "local":
            room = np.where(miss[:, None], global_room, local)
        elif reply_conformity == "blend":
            room = reply_conformity_mix * np.where(
                miss[:, None], global_room, local) + (1.0 - reply_conformity_mix) * global_room
        else:  # "global"
            room = global_room
        stance_new = conv * own_stance + (1.0 - conv) * room + noise

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
