"""The tick engine: wires timing, generation, exposure, reaction, cascades,
perception, and the discourse-state update into the loop spec §3.1 sketches.

Posts persist across ticks (bounded by `post_lifetime`) rather than being
re-exposed within the same tick in recursive waves: each tick's exposure pass
runs over every still-active post, so a repost/quote derived this tick is a
candidate for its own author's followers starting next tick. This is a
simplification of the spec's per-tick cascade sub-loop, made to keep one tick
a single exposure/reaction pass; visibility still decays with `rho ** depth`
regardless of which tick a derived post is exposed in.

Replies spawn posts as of calibration: `derive_posts` handles repost, quote
and reply alike, differing in how far the derived post's stance moves toward
the deriving user (see `dynamics/cascade.py`). What is still not wired is
Hawkes *timing* for those replies — they land in the tick the exposure
happened rather than being scheduled by the open-thread intensity, which
needs `HawkesThreads` bookkeeping across ticks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from discourse_lab.config import Config
from discourse_lab.dynamics.cascade import BRANCHING_ACTIONS, CascadeState, derive_posts, r_eff
from discourse_lab.dynamics.discourse_state import update_discourse
from discourse_lab.dynamics.drift import DriftState, apply_drift
from discourse_lab.dynamics.expression import ExpressionMap
from discourse_lab.dynamics.perception import compute_perception
from discourse_lab.dynamics.posts import PostBatch, concat_post_batches, filter_post_batch, generate_posts
from discourse_lab.dynamics.hawkes import HawkesThreads, generate_reply_posts
from discourse_lab.llm.adjudication import detect_salient_events
from discourse_lab.dynamics.timing import (
    FatigueState,
    circadian_factor,
    circadian_shape,
    sample_post_counts,
)
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
    threads: HawkesThreads = field(init=False)
    circ_shape: np.ndarray = field(init=False)
    phase_ticks: np.ndarray = field(init=False)
    activity: np.ndarray = field(init=False)
    cascade_state: CascadeState = field(default_factory=CascadeState)
    drift_state: DriftState = field(default_factory=DriftState)
    active_posts: PostBatch | None = field(default=None, init=False)
    next_post_id: int = field(default=0, init=False)
    global_stance_var: float = field(init=False)

    # Surfaced for persistence (io/store.py) and for interactive callers who
    # want the raw record off `run_iter` without a writer. Both are replaced
    # every tick — nothing accumulates, so memory stays flat in run length.
    retired_posts: PostBatch | None = field(default=None, init=False)
    engagement_events: dict[str, np.ndarray] | None = field(default=None, init=False)
    exposure_sample: dict[str, np.ndarray] | None = field(default=None, init=False)
    # spec §3.1 step 6: "flag_salient_events(engagements)  # queued, not
    # executed". Channel 3 is the only place the LLM touches dynamics and it
    # is gated. Replaced every tick like the other raw records — accumulating
    # it here and copying the whole list per tick was O(n^2) and broke the
    # flat-memory-in-run-length rule (dev §7.3); the consumer accumulates.
    salient_events: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        n = self.cfg.population.n_users
        K, D = self.cfg.population.n_topics, self.cfg.stance_dims()
        names = self.pop.trait_names

        self.expr = ExpressionMap.build(names, K)
        self.s = np.zeros(K)
        self.sigma = np.zeros((K, D))
        self.fatigue = FatigueState.initial(n)
        self.threads = HawkesThreads()
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
        self.retired_posts = None
        self.engagement_events = None
        self.exposure_sample = None
        self.salient_events = []
        engaged_this_tick: tuple = (None, None, None, None)

        circ = circadian_factor(t, cfg.ticks_per_day, self.phase_ticks, self.circ_shape)
        # posts_per_tick_rate is the Poisson rate at activity = 1 (spec §2.3's
        # lambda_u). It was declared in the config and never applied, so the
        # raw activity trait was the rate: ~2 posts/user/tick instead of
        # ~0.04. Besides the volume, that washed out the heterogeneity the
        # lognormal activity trait exists to create — averaging many Poisson
        # draws per user pulls everyone toward the mean and collapsed the
        # posting-volume Gini.
        n_posts = sample_post_counts(
            rngs["timing"], self.activity * cfg.posts_per_tick_rate, circ, self.fatigue.factor()
        )
        authors = np.repeat(np.arange(n), n_posts)

        new_posts = None
        if len(authors) > 0:
            new_posts = generate_posts(
                authors, self.pop, self.expr, self.s, self.sigma, cfg.trend_eta, rngs["generation"],
                start_id=self.next_post_id, t=t,
            )
            self.next_post_id += len(new_posts)
            self.threads.open_threads(new_posts.id, cfg.hawkes_mu0)
        self.fatigue.step(n_posts, cfg.fatigue_decay)

        if self.active_posts is None:
            self.active_posts = new_posts
        elif new_posts is not None:
            self.active_posts = concat_post_batches([self.active_posts, new_posts])

        n_replies = 0
        # spec §3.1 step 2: replies are drawn from the self-exciting thread
        # intensity (§2.3), not derived from the exposure pass. alpha = ratio
        # * beta keeps the branching ratio alpha/beta = hawkes_ratio < 1.
        reply_targets = self.threads.step(
            rngs["timing"], cfg.hawkes_ratio * cfg.hawkes_beta, cfg.hawkes_beta,
            cfg.max_thread_age, max_replies_per_tick=cfg.max_replies_per_tick,
        )
        if reply_targets:
            reply_posts, reply_warnings = generate_reply_posts(
                reply_targets, self.active_posts, self.pop, self.expr, self.s, self.sigma,
                rngs["generation"], self.next_post_id, t, cfg.max_cascade_depth,
            )
            for w in reply_warnings:
                warnings.warn(w, stacklevel=2)
            if reply_posts is not None:
                self.next_post_id += len(reply_posts)
                # A reply's own thread opens warm in proportion to the thread
                # it landed in — seeded into `excitation`, which decays, never
                # into `mu`, which does not (see HawkesThreads.excitation_of).
                # At 0.0 it opens cold: the spec-literal reading, depth ~1.
                inherited = 0.0
                if cfg.hawkes_mu_inherit > 0:
                    inherited = cfg.hawkes_mu_inherit * self.threads.excitation_of(reply_posts.parent)
                self.threads.open_threads(reply_posts.id, cfg.hawkes_mu0, inherited)
                self.active_posts = concat_post_batches([self.active_posts, reply_posts])
                n_replies = len(reply_posts)

                # Same discipline spec §2.7 applies to cascades via r_eff: the
                # reply process is a branching process too, and it has its own
                # critical point. `hawkes_ratio` bounds excitation *within* a
                # thread, but `hawkes_mu_inherit` adds a second channel across
                # generations, so alpha/beta < 1 alone no longer guarantees
                # stability. Measured at hawkes_ratio=0.6: inherit 0.16 gives
                # 2.4 replies per post, 0.20 gives 473 — the transition is
                # sharp, so warn rather than let a run silently saturate.
                if len(reply_posts) > 50 * max(len(new_posts) if new_posts is not None else 1, 1):
                    warnings.warn(
                        f"hawkes: {len(reply_posts)} replies against "
                        f"{len(new_posts) if new_posts is not None else 0} new posts at t={t} — "
                        f"the reply process looks supercritical. Lower hawkes_mu_inherit "
                        f"({cfg.hawkes_mu_inherit}) or hawkes_ratio ({cfg.hawkes_ratio}).",
                        stacklevel=2,
                    )

        metrics: dict[str, float] = {
            "n_posts": float(len(new_posts) if new_posts is not None else 0),
            "n_replies": float(n_replies),
            "open_threads": 0.0,
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
                    features = compute_features(
                        exposures, posts, self.pop, exposures.is_follower, t,
                        agreement_metric=cfg.agreement_metric,
                    )
                    theta = named_kernel(cfg.kernel)
                    actions = apply_kernel(theta, features, rngs["reaction"])

                    engaged = actions != "skip"
                    before = posts.engagement_count.copy()
                    np.add.at(posts.engagement_count, exposures.post_idx[engaged], 1)

                    # the (user, post, action, t) event log of spec §1.5 —
                    # skips excluded, they are the reference category
                    self.engagement_events = {
                        "t": np.full(int(engaged.sum()), t, dtype=np.int64),
                        "user": exposures.user_id[engaged],
                        "post": posts.id[exposures.post_idx[engaged]],
                        "action": actions[engaged],
                    }

                    cascade_posts, cascade_warnings = derive_posts(
                        actions, exposures.post_idx, exposures.user_id, posts, self.pop, self.expr,
                        self.s, self.sigma, rngs["cascade"], self.cascade_state,
                        cfg.max_cascade_depth, cfg.max_cascade_size, self.next_post_id, t,
                    )
                    for w in cascade_warnings:
                        warnings.warn(w, stacklevel=2)
                    if cascade_posts is not None:
                        self.next_post_id += len(cascade_posts)
                        # Reposts and quotes open threads too: a quote is a
                        # post, and people reply to quotes. They did not, so
                        # every derived post was unreplyable — of 971 depth-1
                        # posts in a 3000-user run, 711 were reposts/quotes
                        # sitting outside the Hawkes pool entirely, which both
                        # understated thread depth and silently made the
                        # branching probability depth-dependent in the wrong
                        # direction.
                        self.threads.open_threads(cascade_posts.id, cfg.hawkes_mu0)
                        self.active_posts = concat_post_batches([self.active_posts, cascade_posts])

                    perceived = compute_perception(n, exposures, exposures.is_follower, posts, self.s, self.sigma)
                    names = self.pop.trait_names
                    stance_cols = [i for i, nm in enumerate(names) if nm.startswith("stance_")]
                    salience, agreement = salience_stance_agreement(perceived, self.pop.X_used[:, stance_cols])

                    reposter_ids = exposures.user_id[np.isin(actions, BRANCHING_ACTIONS)]
                    metrics.update(
                        n_exposures=float(len(exposures)),
                        n_engagements=float(engaged.sum()),
                        attention_gini=attention_gini(posts.engagement_count),
                        salience=salience,
                        agreement=agreement,
                        bubble_index=bubble_index(perceived, self.global_stance_var),
                        r_eff=r_eff(actions, len(exposures), self.graph, reposter_ids),
                    )

                    # spec §3.5: exposures are never persisted in full (they
                    # outnumber engagements ~50:1); a fixed random sample is.
                    if cfg.exposure_sample_rate > 0:
                        keep = rngs["exposure"].random(len(exposures)) < cfg.exposure_sample_rate
                        if keep.any():
                            self.exposure_sample = {
                                "t": np.full(int(keep.sum()), t, dtype=np.int64),
                                "user": exposures.user_id[keep],
                                "post": posts.id[exposures.post_idx[keep]],
                                "rank": exposures.rank[keep],
                                "is_follower": exposures.is_follower[keep],
                                "action": actions[keep],
                            }

                    # spec §3.1 step 6: queued, never executed inside the tick
                    reply_counts = np.zeros(len(posts), dtype=np.int64)
                    np.add.at(reply_counts, exposures.post_idx[actions == "reply"], 1)
                    self.salient_events = list(
                        detect_salient_events(
                            posts, reply_counts,
                            self.cfg.world.adjudication_top_percentile,
                            self.cfg.world.adjudication_pile_on_threshold,
                        )
                    )

                    engaged_this_tick = (posts, posts.engagement_count - before, exposures, actions)

            alive = (t - self.active_posts.t) < cfg.post_lifetime
            if not alive.all():
                # retired posts carry their FINAL engagement count, which is
                # why persistence writes them here rather than at creation
                self.retired_posts = filter_post_batch(self.active_posts, ~alive)
                self.active_posts = filter_post_batch(self.active_posts, alive)

        # spec §3.1 steps 6-7 run EVERY tick, not only on ticks that produced
        # exposures. Both were nested three deep inside the exposure guard.
        # §2.9 is explicit that the OU mean-reversion term "is not optional"
        # and that its absence fails "monotone and quiet" — skipping it on
        # quiet ticks lets drift deltas accumulate without their matching
        # reversion. (Measured, quiet ticks never occur at n_users >= 300, so
        # this was latent rather than active; it is still wrong.)
        metrics["open_threads"] = float(len(self.threads))
        posts_e, delta_e, exposures_e, actions_e = engaged_this_tick

        if posts_e is not None:
            tick_posts = filter_post_batch(posts_e, delta_e > 0)
            tick_posts.engagement_count = delta_e[delta_e > 0]
        else:
            tick_posts = None
        self.s, self.sigma = update_discourse(self.s, self.sigma, tick_posts, cfg.rho_s, cfg.rho_sigma)

        apply_drift(
            self.cfg, self.pop, self.expr, self.drift_state, rngs["drift"], t,
            posts_e, None if delta_e is None else delta_e.astype(float), exposures_e, actions_e,
        )
        self.activity = self.pop.X_used[:, self.pop.trait_names.index("activity")]

        return metrics
