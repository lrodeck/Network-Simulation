"""The tick engine: wires timing, generation, exposure, selection, reaction,
cascades, perception, and the discourse-state update into the loop spec §3.1
sketches.

Posts persist across ticks (bounded by `post_lifetime`) rather than being
re-exposed within the same tick in recursive waves: each tick's exposure pass
runs over every still-active post, so a repost/quote derived this tick is a
candidate for its own author's followers starting next tick. This is a
simplification of the spec's per-tick cascade sub-loop, made to keep one tick
a single exposure/reaction pass; visibility still decays with `rho ** depth`
regardless of which tick a derived post is exposed in.

Replies spawn posts as of calibration: `derive_posts` handles repost, quote
and reply alike, differing in how far the derived post's stance moves toward
the deriving user (see `dynamics/cascade.py`). Reply *timing* follows
`cfg.dynamics.reply_model` (change spec C8): the Hawkes self-exciting draw
(simple contagion, the default) or the distinct-engaged-neighbour threshold
(complex contagion).

Change-spec stages wired here, executed in this order each tick:

    timing/generation   C7 silence gate multiplies the posting rate by the
                        perceived-climate factor (spiral of silence)
    exposure            unchanged: candidate inbox, ranking, budget, decay
    selection           C2.1 content-conditional attention (Bakshy), own
                        phase/stream, between ranking and the kernel
    reaction            C1.2 camp features + C3b per-user kernel gains
    perception          unchanged blend; its output is what C7 gates on
                        NEXT tick (perceived climate, not global state)
    cascade             unchanged
    drift               C1.3 affect_update op, C3a behavior-gradient,
                        C6 repulsion-ablatable channel-2 weights
    rewire              C2.2 slow follow/unfollow on accumulated valence
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
from discourse_lab.dynamics.hawkes import HawkesThreads, generate_reply_posts
from discourse_lab.dynamics.kernel_learning import KernelLearningState, apply_learning
from discourse_lab.dynamics.perception import PerceivedState, compute_perception
from discourse_lab.dynamics.posts import PostBatch, concat_post_batches, filter_post_batch, generate_posts
from discourse_lab.dynamics.reply_model import ThresholdState, threshold_model
from discourse_lab.dynamics.timing import (
    FatigueState,
    circadian_factor,
    circadian_shape,
    sample_post_counts,
    silence_gate_factor,
)
from discourse_lab.exposure import apply_kernel, candidate_inbox, compute_features, named_kernel, rank_candidates
from discourse_lab.exposure.attention import Exposures, select_exposures
from discourse_lab.exposure.kernel import apply_kernel_learned, kernel_with_scales
from discourse_lab.exposure.selection import apply_selection
from discourse_lab.llm.adjudication import detect_salient_events
from discourse_lab.measures import attention_gini, bubble_index, salience_stance_agreement
from discourse_lab.metrics.polarization import camps_and_bimodality
from discourse_lab.network import Graph
from discourse_lab.population import Population
from discourse_lab.dynamics.rewire import RewireState


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
    # C2.2 / C3b / C8 state
    rewire_state: RewireState = field(default_factory=RewireState)
    threshold_state: ThresholdState = field(default_factory=ThresholdState)
    learner: KernelLearningState | None = field(default=None, init=False)
    # C7: users gate on LAST tick's perceived climate — the feed they have
    # already seen, not the one this tick is about to build
    prev_perceived: PerceivedState | None = field(default=None, init=False)
    # C1.2: camp labels + the shared bimodality gate; None when unimodal
    camps: np.ndarray | None = field(default=None, init=False)
    camp_bimodality: float = field(default=float("nan"), init=False)
    active_posts: PostBatch | None = field(default=None, init=False)
    next_post_id: int = field(default=0, init=False)
    global_stance_var: float = field(init=False)

    # Surfaced for persistence (io/store.py) and for interactive callers who
    # want the raw record off `run_iter` without a writer. Both are replaced
    # every tick — nothing accumulates, so memory stays flat in run length.
    retired_posts: PostBatch | None = field(default=None, init=False)
    engagement_events: dict[str, np.ndarray] | None = field(default=None, init=False)
    exposure_sample: dict[str, np.ndarray] | None = field(default=None, init=False)
    # C2.2 edge-change rows for this tick: (t, user, target, added)
    rewire_events: list = field(default_factory=list, init=False)
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

        learning = self.cfg.dynamics.kernel_learning
        if learning not in ("none", "group_gain", "full"):
            raise ValueError(
                f"unknown kernel_learning {learning!r}; expected none | group_gain | full"
            )
        rule = self.cfg.dynamics.kernel_learning_rule
        if rule not in ("conformity", "bandit", "habituation"):
            raise ValueError(
                f"unknown kernel_learning_rule {rule!r}; the learning rule is named in "
                "the config, never implied (change spec C3b)"
            )
        if learning != "none":
            # C3b: kernel-owned state, deliberately NOT a trait-matrix block —
            # the population artifact is keyed independently of the kernel
            self.learner = KernelLearningState(n_users=n)

        self.expr = ExpressionMap.build(
            names, K, quality_trait_coupling=self.cfg.dynamics.quality_trait_coupling
        )
        self.s = np.zeros(K)
        self.sigma = np.zeros((K, D))
        self.fatigue = FatigueState.initial(n)
        self.threads = HawkesThreads()
        self.circ_shape = circadian_shape(self.cfg.dynamics.ticks_per_day)

        circadian_phase = self.pop.X_used[:, names.index("circadian_phase")]
        self.phase_ticks = ((circadian_phase + np.pi) / (2 * np.pi)) * self.cfg.dynamics.ticks_per_day

        self.activity = self.pop.X_used[:, names.index("activity")]

        stance_cols = [i for i, name in enumerate(names) if name.startswith("stance_")]
        self.stance_cols = stance_cols
        self.global_stance_var = float(self.pop.X_used[:, stance_cols].var()) if stance_cols else 1.0

    def _refresh_camps(self) -> None:
        """C1.2: camp labels under the shared bimodality gate. Recomputed per
        tick because stance drifts; None (not zeros) when the population is
        unimodal — camp is then noise and every camp-conditional number is
        undefined rather than zero (metrics/polarization.py)."""
        if len(self.stance_cols) == 0:
            self.camps = None
            return
        stance = self.pop.X_used[:, self.stance_cols]
        self.camps, self.camp_bimodality = camps_and_bimodality(stance)

    def step(self, t: int) -> dict[str, float]:
        cfg = self.cfg.dynamics
        rngs = self.rngs
        n = self.cfg.population.n_users
        self.retired_posts = None
        self.engagement_events = None
        self.exposure_sample = None
        self.salient_events = []
        self.rewire_events = []
        engaged_this_tick: tuple = (None, None, None, None)

        circ = circadian_factor(t, cfg.ticks_per_day, self.phase_ticks, self.circ_shape)
        # posts_per_tick_rate is the Poisson rate at activity = 1 (spec §2.3's
        # lambda_u). It was declared in the config and never applied, so the
        # raw activity trait was the rate: ~2 posts/user/tick instead of
        # ~0.04. Besides the volume, that washed out the heterogeneity the
        # lognormal activity trait exists to create — averaging many Poisson
        # draws per user pulls everyone toward the mean and collapsed the
        # posting-volume Gini.
        #
        # C7: the spiral-of-silence gate multiplies the rate by the perceived
        # climate factor. It gates WHETHER one posts, never what a post says
        # — which is why expressed stance can diverge from latent stance
        # (false consensus) only when the gate binds.
        gate = None
        if cfg.silence_gate > 0:
            if self.prev_perceived is not None:
                gate = silence_gate_factor(
                    self.pop.X_used[:, self.stance_cols],
                    self.prev_perceived,
                    self.pop.X_used[:, self.pop.trait_names.index("conviction")],
                    cfg.silence_gate,
                    cfg.silence_conviction_moderation,
                )
            else:
                # no perceived climate yet (tick 0): the gate cannot bind on
                # a feed nobody has seen, so it is exactly 1
                gate = np.ones(n)
        n_posts = sample_post_counts(
            rngs["timing"], self.activity * cfg.posts_per_tick_rate, circ, self.fatigue.factor(),
            gate=gate,
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
        #
        # C8: the reply model is a registry choice, not a hard-coded theory —
        # "hawkes" is simple contagion (this draw), "threshold" replaces the
        # intensity draw with the distinct-engaged-neighbour rule.
        reply_targets: dict[int, int] = {}
        reply_authors: dict[int, np.ndarray] = {}
        if cfg.reply_model == "threshold":
            self.threshold_state.prune(t, cfg.max_thread_age)
            reply_targets, reply_authors = threshold_model(
                state=self.threshold_state,
                graph=self.graph,
                active_posts=self.active_posts,
                reply_prop=self.pop.X_used[:, self.pop.trait_names.index("reply_prop")],
                rng=rngs["timing"],
                max_age=cfg.max_thread_age,
                max_replies_per_tick=cfg.max_replies_per_tick,
                threshold_scale=cfg.threshold_scale,
            )
            # thread bookkeeping still ages; only the intensity DRAW is
            # replaced by the threshold rule
            self.threads.step(rngs["timing"], 0.0, cfg.hawkes_beta, cfg.max_thread_age, draw=False)
        else:
            reply_targets = self.threads.step(
                rngs["timing"], cfg.hawkes_ratio * cfg.hawkes_beta, cfg.hawkes_beta,
                cfg.max_thread_age, max_replies_per_tick=cfg.max_replies_per_tick,
            )
        if reply_targets:
            reply_posts, reply_warnings = generate_reply_posts(
                reply_targets, self.active_posts, self.pop, self.expr, self.s, self.sigma,
                rngs["generation"], self.next_post_id, t, cfg.max_cascade_depth,
                authors_for_target=reply_authors or None,
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
                        f"replies: {len(reply_posts)} replies against "
                        f"{len(new_posts) if new_posts is not None else 0} new posts at t={t} — "
                        f"the reply process looks supercritical (reply_model={cfg.reply_model}). "
                        f"Lower hawkes_mu_inherit ({cfg.hawkes_mu_inherit}) or hawkes_ratio ({cfg.hawkes_ratio}).",
                        stacklevel=2,
                    )

        metrics: dict[str, float] = {
            "n_posts": float(len(new_posts) if new_posts is not None else 0),
            "n_replies": float(n_replies),
            "open_threads": 0.0,
            "n_exposures": 0.0,
            "n_attended": 0.0,
            "n_engagements": 0.0,
            "attention_gini": float("nan"),
            "salience": float("nan"),
            "agreement": float("nan"),
            "bubble_index": float("nan"),
            "r_eff": 0.0,
            "camp_bimodality": float(self.camp_bimodality) if np.isfinite(self.camp_bimodality) else float("nan"),
        }

        self._refresh_camps()

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
                    metrics["n_exposures"] = float(len(exposures))
                    features = compute_features(
                        exposures, posts, self.pop, exposures.is_follower, t,
                        agreement_metric=cfg.agreement_metric,
                        camps=self.camps,
                    )

                    # C2.1: the selection stage — content-conditional
                    # attention between ranking and the kernel, in its own
                    # phase stream. Only the ATTENDED subset reaches the
                    # kernel; the exposure log records both states.
                    attended = apply_selection(
                        cfg.selection, exposures, features, cfg.tau_position,
                        rngs["selection"], cfg.selection_beta,
                    )
                    n_attended = int(attended.sum())
                    metrics["n_attended"] = float(n_attended)
                    exposures_att = Exposures(
                        post_idx=exposures.post_idx[attended],
                        user_id=exposures.user_id[attended],
                        rank=exposures.rank[attended],
                        is_follower=exposures.is_follower[attended],
                    )
                    features_att = {k: v[attended] for k, v in features.items()}

                    if n_attended > 0:
                        theta = kernel_with_scales(cfg.kernel, cfg.kernel_theta, cfg.theta_scale)
                        if self.learner is not None:
                            # C3b: per-user gains modulate the named kernel;
                            # g == 1 reproduces it exactly
                            self.learner.ensure_initialized()
                            gains_rows = self.learner.g[exposures_att.user_id]
                            actions = apply_kernel_learned(theta, features_att, gains_rows, rngs["reaction"])
                        else:
                            actions = apply_kernel(theta, features_att, rngs["reaction"])

                        engaged = actions != "skip"
                        before = posts.engagement_count.copy()
                        np.add.at(posts.engagement_count, exposures_att.post_idx[engaged], 1)
                        engagement_delta = posts.engagement_count - before

                        # the (user, post, action, t) event log of spec §1.5 —
                        # skips excluded, they are the reference category
                        self.engagement_events = {
                            "t": np.full(int(engaged.sum()), t, dtype=np.int64),
                            "user": exposures_att.user_id[engaged],
                            "post": posts.id[exposures_att.post_idx[engaged]],
                            "action": actions[engaged],
                        }

                        # C8 threshold model: accumulate distinct engagers per
                        # post from the event log it will draw on next tick
                        if cfg.reply_model == "threshold":
                            self.threshold_state.observe(
                                t, exposures_att.user_id[engaged],
                                posts.id[exposures_att.post_idx[engaged]], actions[engaged],
                            )

                        cascade_posts, cascade_warnings = derive_posts(
                            actions, exposures_att.post_idx, exposures_att.user_id, posts, self.pop, self.expr,
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

                        engaged_this_tick = (posts, engagement_delta, exposures_att, actions, features_att)

                    # C2.1: the 1% diagnostic sample now spans the EXPOSED set
                    # with an `attended` flag, so the echo-chamber index can be
                    # computed on either quantity without ambiguity (change
                    # spec C2.1: conflating them is the failure to avoid).
                    # Unattended rows carry action="unattended" — they were
                    # never offered to the kernel, so they are neither skips
                    # nor engagements.
                    if cfg.exposure_sample_rate > 0:
                        keep = rngs["exposure"].random(len(exposures)) < cfg.exposure_sample_rate
                        if keep.any():
                            kept_idx = np.flatnonzero(keep)
                            att_rows = np.flatnonzero(attended)
                            pos_in_att = np.clip(
                                np.searchsorted(att_rows, kept_idx), 0, max(len(att_rows) - 1, 0)
                            )
                            # only read where attended: unattended positions are
                            # insertion points, not real indices
                            sample_actions = np.where(
                                attended[kept_idx], actions[pos_in_att], "unattended"
                            )
                            self.exposure_sample = {
                                "t": np.full(len(kept_idx), t, dtype=np.int64),
                                "user": exposures.user_id[kept_idx],
                                "post": posts.id[exposures.post_idx[kept_idx]],
                                "rank": exposures.rank[kept_idx],
                                "is_follower": exposures.is_follower[kept_idx],
                                "attended": attended[kept_idx],
                                "action": sample_actions,
                            }

                    if n_attended > 0:
                        perceived = compute_perception(n, exposures_att, exposures_att.is_follower, posts, self.s, self.sigma)
                        self.prev_perceived = perceived
                        salience, agreement = salience_stance_agreement(perceived, self.pop.X_used[:, self.stance_cols])

                        reposter_ids = exposures_att.user_id[np.isin(actions, BRANCHING_ACTIONS)]
                        metrics.update(
                            n_engagements=float(engaged.sum()),
                            attention_gini=attention_gini(posts.engagement_count),
                            salience=salience,
                            agreement=agreement,
                            bubble_index=bubble_index(perceived, self.global_stance_var),
                            r_eff=r_eff(actions, n_attended, self.graph, reposter_ids),
                        )

                        # spec §3.1 step 6: queued, never executed inside the tick
                        reply_counts = np.zeros(len(posts), dtype=np.int64)
                        np.add.at(reply_counts, exposures_att.post_idx[actions == "reply"], 1)
                        self.salient_events = list(
                            detect_salient_events(
                                posts, reply_counts,
                                self.cfg.world.adjudication_top_percentile,
                                self.cfg.world.adjudication_pile_on_threshold,
                            )
                        )
                else:
                    metrics["n_exposures"] = 0.0

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
        posts_e, delta_e, exposures_e, actions_e, features_e = engaged_this_tick

        if posts_e is not None:
            tick_posts = filter_post_batch(posts_e, delta_e > 0)
            tick_posts.engagement_count = delta_e[delta_e > 0]
        else:
            tick_posts = None
        self.s, self.sigma = update_discourse(self.s, self.sigma, tick_posts, cfg.rho_s, cfg.rho_sigma)

        apply_drift(
            self.cfg, self.pop, self.expr, self.drift_state, rngs["drift"], t,
            posts_e, None if delta_e is None else delta_e.astype(float), exposures_e, actions_e,
            camps=self.camps,
        )
        self.activity = self.pop.X_used[:, self.pop.trait_names.index("activity")]

        # C3b: kernel learning runs after the tick's own events exist to learn
        # from; g is read by NEXT tick's reaction pass
        if self.learner is not None:
            if exposures_e is not None and len(exposures_e) > 0:
                apply_learning(
                    self.learner, self.cfg, self.graph, exposures_e, actions_e, posts_e,
                    features_e, delta_e.astype(float), t, rngs["reaction"],
                )
            metrics["kernel_gain_dev"] = float(np.abs(self.learner.g - 1.0).mean()) if self.learner.g is not None else 0.0

        # C2.2: accumulate this tick's interaction valence, then maybe rewire
        if posts_e is not None and delta_e is not None and len(delta_e) > 0:
            engaged_mask = np.isin(actions_e, ("like", "repost", "reply", "quote", "report"))
            if engaged_mask.any():
                self.rewire_state.observe(
                    exposures_e.user_id[engaged_mask],
                    posts_e.author[exposures_e.post_idx[engaged_mask]],
                    actions_e[engaged_mask],
                )
        if self.rewire_state.maybe_rewire(t, self.graph, self.cfg, rngs["rewire"]):
            self.rewire_events = list(self.rewire_state.events)
            self.rewire_state.events = []

        return metrics
