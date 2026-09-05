"""The tick engine: wires timing, generation, exposure, reaction, cascades,
perception, and the discourse-state update into the loop spec §3.1 sketches.

Posts persist across ticks (bounded by `post_lifetime`) rather than being
re-exposed within the same tick in recursive waves: each tick's exposure pass
runs over every still-active post, so a repost/quote derived this tick is a
candidate for its own author's followers starting next tick. This is a
simplification of the spec's per-tick cascade sub-loop, made to keep one tick
a single exposure/reaction pass; visibility still decays with `rho ** depth`
regardless of which tick a derived post is exposed in.

Also not yet wired here: Hawkes-driven reply *posts* (a "reply" action is
counted as an engagement and feeds the discourse-state update, but does not
yet spawn a new reply PostBatch entry — that needs open-thread bookkeeping
tied to `HawkesThreads` across ticks, left for a follow-up). Drift (step 11)
is not implemented yet either, so traits are static within a run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from discourse_lab.config import Config
from discourse_lab.dynamics.cascade import CascadeState, derive_posts, r_eff
from discourse_lab.dynamics.discourse_state import update_discourse
from discourse_lab.dynamics.drift import DriftState, apply_drift
from discourse_lab.dynamics.expression import ExpressionMap
from discourse_lab.dynamics.perception import compute_perception
from discourse_lab.dynamics.posts import PostBatch, concat_post_batches, filter_post_batch, generate_posts
from discourse_lab.dynamics.timing import FatigueState, circadian_factor, circadian_shape, sample_post_counts
from discourse_lab.exposure import apply_kernel, candidate_inbox, compute_features, named_kernel, rank_candidates
from discourse_lab.exposure.attention import select_exposures
from discourse_lab.measures import attention_gini, bubble_index, salience_stance_agreement
from discourse_lab.network import Graph
from discourse_lab.population import Population


@dataclass
class TickEngine:
    cfg: Config
    pop: Population
    graph: Graph
    rngs: dict[str, np.random.Generator]

    expr: ExpressionMap = field(init=False)
    s: np.ndarray = field(init=False)
    sigma: np.ndarray = field(init=False)
    fatigue: FatigueState = field(init=False)
    circ_shape: np.ndarray = field(init=False)
    phase_ticks: np.ndarray = field(init=False)
    activity: np.ndarray = field(init=False)
    cascade_state: CascadeState = field(default_factory=CascadeState)
    drift_state: DriftState = field(default_factory=DriftState)
    active_posts: PostBatch | None = field(default=None, init=False)
    next_post_id: int = field(default=0, init=False)
    global_stance_var: float = field(init=False)

    def __post_init__(self) -> None:
        n = self.cfg.population.n_users
        K, D = self.cfg.population.n_topics, self.cfg.stance_dims()
        names = self.pop.trait_names

        self.expr = ExpressionMap.build(names, K)
        self.s = np.zeros(K)
        self.sigma = np.zeros((K, D))
        self.fatigue = FatigueState.initial(n)
        self.circ_shape = circadian_shape(self.cfg.dynamics.ticks_per_day)

        circadian_phase = self.pop.X_used[:, names.index("circadian_phase")]
        self.phase_ticks = ((circadian_phase + np.pi) / (2 * np.pi)) * self.cfg.dynamics.ticks_per_day

        self.activity = self.pop.X_used[:, names.index("activity")]

        stance_cols = [i for i, name in enumerate(names) if name.startswith("stance_")]
        self.global_stance_var = float(self.pop.X_used[:, stance_cols].var()) if stance_cols else 1.0

    def step(self, t: int) -> dict[str, float]:
        cfg = self.cfg.dynamics
        rngs = self.rngs
        n = self.cfg.population.n_users

        circ = circadian_factor(t, cfg.ticks_per_day, self.phase_ticks, self.circ_shape)
        n_posts = sample_post_counts(rngs["timing"], self.activity, circ, self.fatigue.factor())
        authors = np.repeat(np.arange(n), n_posts)

        new_posts = None
        if len(authors) > 0:
            new_posts = generate_posts(
                authors, self.pop, self.expr, self.s, self.sigma, cfg.trend_eta, rngs["generation"],
                start_id=self.next_post_id, t=t,
            )
            self.next_post_id += len(new_posts)
        self.fatigue.step(n_posts, cfg.fatigue_decay)

        if self.active_posts is None:
            self.active_posts = new_posts
        elif new_posts is not None:
            self.active_posts = concat_post_batches([self.active_posts, new_posts])

        metrics: dict[str, float] = {
            "n_posts": float(len(new_posts) if new_posts is not None else 0),
            "n_exposures": 0.0,
            "n_engagements": 0.0,
            "attention_gini": float("nan"),
            "salience": float("nan"),
            "agreement": float("nan"),
            "bubble_index": float("nan"),
            "r_eff": 0.0,
        }

        if self.active_posts is not None and len(self.active_posts) > 0:
            posts = self.active_posts
            pairs = candidate_inbox(self.graph, posts, cfg.inject_k, self.cfg.graph.fanout_cap, rngs["exposure"])

            if len(pairs) > 0:
                scores = rank_candidates(cfg.ranker, pairs, posts, self.pop, rngs["exposure"])
                exposures = select_exposures(
                    pairs, scores, self.activity, cfg.attention_budget, cfg.tau_position, rngs["exposure"],
                    cascade_depth=posts.depth[pairs.post_idx], cascade_rho=cfg.cascade_depth_decay,
                )

                if len(exposures) > 0:
                    features = compute_features(exposures, posts, self.pop, exposures.is_follower, t)
                    theta = named_kernel(cfg.kernel)
                    actions = apply_kernel(theta, features, rngs["reaction"])

                    engaged = actions != "skip"
                    before = posts.engagement_count.copy()
                    np.add.at(posts.engagement_count, exposures.post_idx[engaged], 1)

                    cascade_posts, cascade_warnings = derive_posts(
                        actions, exposures.post_idx, exposures.user_id, posts, self.pop, self.expr,
                        self.s, self.sigma, rngs["cascade"], self.cascade_state,
                        cfg.max_cascade_depth, cfg.max_cascade_size, self.next_post_id, t,
                    )
                    for w in cascade_warnings:
                        warnings.warn(w, stacklevel=2)
                    if cascade_posts is not None:
                        self.next_post_id += len(cascade_posts)
                        self.active_posts = concat_post_batches([self.active_posts, cascade_posts])

                    perceived = compute_perception(n, exposures, exposures.is_follower, posts, self.s, self.sigma)
                    names = self.pop.trait_names
                    stance_cols = [i for i, nm in enumerate(names) if nm.startswith("stance_")]
                    salience, agreement = salience_stance_agreement(perceived, self.pop.X_used[:, stance_cols])

                    reposter_ids = exposures.user_id[np.isin(actions, ("repost", "quote"))]
                    metrics.update(
                        n_exposures=float(len(exposures)),
                        n_engagements=float(engaged.sum()),
                        attention_gini=attention_gini(posts.engagement_count),
                        salience=salience,
                        agreement=agreement,
                        bubble_index=bubble_index(perceived, self.global_stance_var),
                        r_eff=r_eff(actions, len(exposures), self.graph, reposter_ids),
                    )

                    delta = posts.engagement_count - before
                    tick_posts = filter_post_batch(posts, delta > 0)
                    tick_posts.engagement_count = delta[delta > 0]
                    self.s, self.sigma = update_discourse(self.s, self.sigma, tick_posts, cfg.rho_s, cfg.rho_sigma)

                    apply_drift(
                        self.cfg, self.pop, self.expr, self.drift_state, rngs["drift"], t,
                        posts, delta.astype(float), exposures, actions,
                    )
                    self.activity = self.pop.X_used[:, self.pop.trait_names.index("activity")]

            alive = (t - self.active_posts.t) < cfg.post_lifetime
            self.active_posts = filter_post_batch(self.active_posts, alive) if not alive.all() else self.active_posts

        return metrics
