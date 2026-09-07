"""Trait drift (spec §2.9): two free channels plus Ornstein-Uhlenbeck
mean-reversion. Operates on stored (unconstrained) traits, so it is a plain
additive process that can never leave the feasible set (spec §1.1).

Channel 3 (LLM adjudication) is queued only — gated, event-triggered, and
deferred to step 12; `llm_adjudication` in config stays unused here.

Change-spec additions, all composed additively into the same gain matrix so
they appear together in the per-op contribution-norm diagnostic
(`DriftState.op_norms`, plotted by the run monitor):

- **C1.3 `affect_update`** — the affect block (identification / animus) is a
  *state*, not a fixed trait: it updates from interaction outcomes. It uses
  its own weight tables, deliberately separate from channel 2's, because
  replying to an out-group post is *engagement* with the out-group
  (animus-raising) while it is stance-repulsive in channel 2. Conflating the
  two tables is the single most likely implementation error here.
- **C3a** — channel 1's gradient generalized from the expression block to
  every trait column with a generation-map path, including the behavior
  columns (activity, reply_prop, repost_prop). Brady, McLoughlin, Doan &
  Crockett (Science Advances 2021): outrage expression is socially learned;
  a kernel that encodes outrage as a fixed disposition cannot produce norm
  convergence. `contrarianism` is deliberately NOT in that set — it governs
  engagement *choice*, not post *content*, so it has no residual and is
  handled by C3b's kernel learning instead.
- **C6** — channel 2's weight table is split into attractive and repulsive
  halves with `repulsion` as an ablatable switch (Mäs & Flache 2013;
  Takács et al. 2016 show the repulsive-influence assumption has mixed,
  hard-to-identify support). `repulsion=False` zeroes the negative weights
  outright; it does NOT clamp deltas to non-negative — clamping introduces a
  rectification nonlinearity, which is a different model, not an ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from discourse_lab.config import Config
from discourse_lab.dynamics.expression import POST_DIM_LINKS, POST_DIMS, ExpressionMap
from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.exposure.attention import Exposures
from discourse_lab.population import Population
from discourse_lab.population.links import to_stored, to_used
from discourse_lab.population.traits import BEHAVIOR, EXPRESSION, trait_table
from discourse_lab.registry import get, names, register

# Per-block reversion rate (spec §2.9): expression reverts fast (style is
# fashion), stance reverts slowly, personality effectively not at all.
# C1.3: affect sits between them (change spec: "stickier than style, less
# sticky than position") — a guess, not a calibration; it is flagged in
# MODEL.md §15 and carried in the C10 sensitivity sweep.
DEFAULT_K: dict[str, float] = {
    "personality": 0.0,
    "expression": 0.05,
    "topic_affinity": 0.02,
    "stance": 0.005,
    "behavior": 0.01,
    "meta": 0.01,
    "affect": 0.02,
}

# Channel 2 action weights (spec §2.9), split per change spec C6 into the
# attractive and repulsive halves. `quote` is not in the spec's table;
# treated as a weaker rebroadcast than a plain repost since it carries the
# quoter's own (possibly critical) commentary. These module defaults are the
# *shipped* values; the live tables come from `cfg.dynamics`
# (`social_weights_positive` / `social_weights_negative` / `repulsion`) so
# the repulsion assumption is ablatable rather than load-bearing-and-
# invisible.
ACTION_WEIGHTS: dict[str, float] = {
    "like": 1.0,
    "repost": 1.5,
    "quote": 0.5,
    "reply": -0.5,
    "report": -2.0,
    "skip": 0.0,
}

# C1.3: affect weight tables — the SHIPPED DEFAULTS. The live tables come
# from `cfg.dynamics` (`affect_weights_hostility` / `affect_weights_support`),
# config-side for the same reason C6 moved channel 2's weights there: the
# hate-engagement reading (any out-group engagement raises animus, scaled by
# how confrontational the action is — Rathje et al. 2021) is a theory, and
# the contact-hypothesis alternative (likes as positive contact that LOWERS
# animus) is a different theory a sweep should be able to express without
# editing the loop. Deliberately NOT channel 2's table: replying to an
# out-group post is engagement with the out-group (animus-raising) while it
# is stance-repulsive in channel 2, and conflating the two tables is the
# single most likely implementation error here. The change spec's summary
# line lists reply at -0.5, channel 2's value; its own rationale states the
# opposite sign, and the rationale is the load-bearing part.
AFFECT_HOSTILITY_WEIGHTS: dict[str, float] = {
    "like": 0.25,      # fleeting out-group contact
    "repost": 0.25,    # amplifying out-group content
    "quote": 0.5,      # quoting is engagement carrying critique
    "reply": 0.5,      # a direct cross-camp exchange
    "report": 2.0,     # the hostile act
    "skip": 0.0,       # not attending is not an interaction outcome
}

# C1.3: the identification table mirrors it — in-group support raises
# attachment to the camp; hostile acts against in-group content erode it.
AFFECT_SUPPORT_WEIGHTS: dict[str, float] = {
    "like": 1.0,
    "repost": 1.5,
    "quote": 0.5,
    "reply": 0.5,
    "report": -1.0,
    "skip": 0.0,
}

# C3a: behavior columns with a generation-map path — the only behavior
# traits channel 1's generalized gradient may touch. `contrarianism` and
# `credulity` govern engagement choice, not post content, and are excluded;
# `prominence` is an ascribed property, not a learned behaviour.
BEHAVIOR_REINFORCE_COLS = ("activity", "reply_prop", "repost_prop")


@dataclass
class DriftState:
    """`Bs`, the slow-moving mean-reversion target, initialised to `X_stored`
    the first time drift runs (spec §2.9's `Bs` starts at the population's
    own baseline, not zero). `engagement_baseline` is a per-user running
    expectation of their own engagement, needed to make "surprise" mean
    anything for an author who only posted once this tick (spec §2.9's
    `E[engagement | author_p]` reads as a standing expectation, not
    something recomputable from a single observation).

    `op_norms` carries the per-op contribution norm of the last tick — the
    cancellation diagnostic the run monitor plots (dev §7.1).
    """

    Bs: np.ndarray | None = field(default=None)
    engagement_baseline: np.ndarray | None = field(default=None)
    baseline_ema: float = 0.2
    op_norms: dict[str, float] = field(default_factory=dict)

    def ensure_initialized(self, x_stored: np.ndarray) -> None:
        if self.Bs is None:
            self.Bs = x_stored.copy()
        if self.engagement_baseline is None:
            self.engagement_baseline = np.zeros(x_stored.shape[0])

    def surprise(self, author: np.ndarray, engagement_delta: np.ndarray) -> np.ndarray:
        """`r_p = engagement_p - E[engagement | author_p]`, then rolls this
        tick's per-author mean into the running baseline (EMA) for next time.
        """
        n = len(self.engagement_baseline)
        surprise = engagement_delta - self.engagement_baseline[author]

        per_author_sum = np.zeros(n)
        counts = np.zeros(n)
        np.add.at(per_author_sum, author, engagement_delta)
        np.add.at(counts, author, 1.0)
        posted = counts > 0
        per_author_mean = np.divide(per_author_sum, counts, out=np.zeros(n), where=posted)
        self.engagement_baseline[posted] = (
            (1 - self.baseline_ema) * self.engagement_baseline[posted] + self.baseline_ema * per_author_mean[posted]
        )
        return surprise


def block_rates(cfg: Config, trait_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (k, k_b) reversion rates from `DEFAULT_K`, overridden by
    `cfg.dynamics.ou_k`. `k_b ~= k / 10` (spec §2.9). The affect block's base
    rate comes from `cfg.dynamics.affect_ou_k` (C1.3) unless `ou_k`
    overrides the block explicitly.
    """
    specs = trait_table(cfg)
    assert [s.name for s in specs] == trait_names
    overrides = dict(cfg.dynamics.ou_k)

    k = np.array([overrides.get(s.block, DEFAULT_K[s.block]) for s in specs])
    # C1.3: the config dial, applied below any explicit ou_k override
    if cfg.population.affect and "affect" not in overrides:
        k[[s.block == "affect" for s in specs]] = cfg.dynamics.affect_ou_k
    return k, k / 10.0


def ramp_factor(t: int, ramp_ticks: int) -> float:
    """Gains ramped linearly from 0 to 1 over `ramp_ticks`, then held at 1 —
    dev §6 step 11: bring each channel up gradually rather than switching it
    on at full strength from tick 0.
    """
    if ramp_ticks <= 0:
        return 1.0
    return min(1.0, t / ramp_ticks)


def reinforcement_delta(
    posts: PostBatch,
    pop: Population,
    expr: ExpressionMap,
    surprise: np.ndarray,
) -> np.ndarray:
    """Channel 1 (spec §2.9), generalized per change spec C3a: a bandit-style
    pull of each author's traits toward whatever they posted when it did
    better than expected — over every trait column with a generation-map
    path, not just expression.

    `surprise` (`r_p`, from `DriftState.surprise`) weights each post's
    contribution. The post's *predicted* style, `X_stored[author] @ A.T`, is
    the expression map's own trait->post-dims map; the actual style is
    recovered by inverting each dim's link on the post's used-space value.
    Moving traits by `A^T @ residual` is exactly the gradient of that linear
    map — the direction that would have produced more of the engaging
    residual. Expression columns learn style (§2.9 channel 1 as written);
    behavior columns (`BEHAVIOR_REINFORCE_COLS`) learn *propensity* —
    Brady et al. (2021)'s socially-learned outrage, C3a. Columns without an
    A path (contrarianism, credulity, prominence, ...) get exactly zero:
    there is no residual for them to learn from, and inventing a
    pseudo-residual for an engagement-choice trait would be a different
    mechanism (C3b's kernel learning is where choice lives).

    Averaged (not summed) over an author's posts this tick, so the step size
    is set by `drift_lr` alone rather than incidentally scaling with how
    many times a busy author happened to post. Returns the full-width
    (n, n_traits) gradient; the caller applies the per-block learning rates.
    """
    n = pop.X_stored.shape[0]
    names = pop.trait_names
    delta = np.zeros((n, len(names)))
    if len(posts) == 0:
        return delta

    author = posts.author
    actual_stored = np.stack([to_stored(getattr(posts, dim), POST_DIM_LINKS[dim]) for dim in POST_DIMS], axis=1)
    predicted_stored = pop.X_stored[author] @ expr.A.T
    residual = actual_stored - predicted_stored  # (M, |POST_DIMS|)

    contribution = surprise[:, None] * (residual @ expr.A)  # (M, n_traits)

    counts = np.zeros(n)
    np.add.at(counts, author, 1.0)
    np.add.at(delta, author, contribution)
    delta[counts > 0] /= counts[counts > 0, None]
    return delta


def _action_weights(actions: np.ndarray, table: dict[str, float]) -> np.ndarray:
    keys = np.array(sorted(table))
    vals = np.array([table[k] for k in keys])
    idx = np.searchsorted(keys, actions)
    idx = np.clip(idx, 0, len(keys) - 1)
    hit = keys[idx] == actions
    return np.where(hit, vals[idx], 0.0)


def social_weights(cfg: Config) -> dict[str, float]:
    """Channel 2's live table (C6): the attractive half always applies; the
    repulsive half is zeroed when `repulsion=False`. Zeroing the *weights*
    (not clamping the deltas) keeps the update linear — clamping would be a
    rectification nonlinearity, which is a different model, not an ablation.
    """
    table = dict(cfg.dynamics.social_weights_positive)
    table["skip"] = 0.0
    if cfg.dynamics.repulsion:
        table.update(dict(cfg.dynamics.social_weights_negative))
    else:
        table.update({action: 0.0 for action, _ in cfg.dynamics.social_weights_negative})
    return table


def social_influence_delta(
    exposures: Exposures,
    actions: np.ndarray,
    posts: PostBatch,
    pop: Population,
    stance_cols: list[int],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Channel 2 (spec §2.9): exposure-weighted pull of stance toward what a
    user consumed and did not reject. `posts` is whichever batch
    `exposures.post_idx` indexes into.

    Averaged (not summed) over a user's exposures this tick, for the same
    reason as channel 1: the step size should be set by `drift_lr_social`,
    not by how many posts a heavy feed happened to serve this tick.
    `weights` overrides the action table (C6: the repulsion ablation swaps
    it without touching this function).
    """
    n = pop.X_used.shape[0]
    d = len(stance_cols)
    delta = np.zeros((n, d))
    if len(exposures) == 0:
        return delta

    # vectorised lookup: a list comprehension here is a per-exposure Python
    # loop over tens of thousands of actions each tick (spec §0.5)
    table = weights if weights is not None else ACTION_WEIGHTS
    w = _action_weights(actions, table)
    user_stance = pop.X_used[exposures.user_id][:, stance_cols]
    post_stance = posts.stance[exposures.post_idx]
    contribution = w[:, None] * (post_stance - user_stance)

    counts = np.zeros(n)
    np.add.at(counts, exposures.user_id, 1.0)
    np.add.at(delta, exposures.user_id, contribution)
    delta[counts > 0] /= counts[counts > 0, None]
    return delta


def affect_delta(
    exposures: Exposures,
    actions: np.ndarray,
    posts: PostBatch,
    pop: Population,
    camps: np.ndarray | None,
    lr_affect: float,
    hostility_weights: dict[str, float] | None = None,
    support_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """C1.3 `affect_update`, registered as a drift op below so it composes
    additively with the other channels and shows up in the op-norm diagnostic.

        Δanimus_u         = lr_affect · mean_over_exposures(
                                outgroup · hostility_weight(action) )
        Δidentification_u = lr_affect · mean_over_exposures(
                                ingroup  · support_weight(action) )

    Weight tables default to `AFFECT_HOSTILITY_WEIGHTS` /
    `AFFECT_SUPPORT_WEIGHTS`; the live tables come from `cfg.dynamics` via
    `affect_weights(cfg)` — separate from channel 2's, see the module
    docstring. Returns the full-width (n, n_traits) gain block, zero outside
    the affect columns.
    """
    n = pop.X_stored.shape[0]
    names = pop.trait_names
    affect_cols = [i for i, nm in enumerate(names) if nm in ("identification", "animus")]
    delta = np.zeros((n, len(names)))
    if camps is None or len(exposures) == 0 or not pop.has_affect or not affect_cols:
        return delta

    u = exposures.user_id
    outgroup = (camps[u] != camps[posts.author[exposures.post_idx]]).astype(float)
    ingroup = 1.0 - outgroup

    host_w = _action_weights(actions, hostility_weights or AFFECT_HOSTILITY_WEIGHTS)
    supp_w = _action_weights(actions, support_weights or AFFECT_SUPPORT_WEIGHTS)

    counts = np.zeros(n)
    animus_acc = np.zeros(n)
    ident_acc = np.zeros(n)
    np.add.at(counts, u, 1.0)
    np.add.at(animus_acc, u, lr_affect * outgroup * host_w)
    np.add.at(ident_acc, u, lr_affect * ingroup * supp_w)

    animus_col = names.index("animus")
    ident_col = names.index("identification")
    mean_animus = np.divide(animus_acc, counts, out=np.zeros(n), where=counts > 0)
    mean_ident = np.divide(ident_acc, counts, out=np.zeros(n), where=counts > 0)
    delta[:, animus_col] = mean_animus
    delta[:, ident_col] = mean_ident
    return delta


@register("drift_op", "affect_update")
def affect_update(ctx: dict) -> np.ndarray:
    """Registry face of C1.3 (change spec §0.3): composition stays additive
    and the op is addressable by name from configs and the run monitor."""
    return affect_delta(
        ctx["exposures"], ctx["actions"], ctx["posts"], ctx["pop"],
        ctx["camps"], ctx["lr_affect"],
        hostility_weights=ctx.get("hostility_weights"),
        support_weights=ctx.get("support_weights"),
    )


def affect_weights(cfg: Config) -> tuple[dict[str, float], dict[str, float]]:
    """The live C1.3 tables from `cfg.dynamics`; `skip` is structurally 0."""
    hostility = dict(cfg.dynamics.affect_weights_hostility)
    support = dict(cfg.dynamics.affect_weights_support)
    hostility["skip"] = 0.0
    support["skip"] = 0.0
    return hostility, support


def drift_op_names() -> list[str]:
    return names("drift_op")


def apply_drift(
    cfg: Config,
    pop: Population,
    expr: ExpressionMap,
    state: DriftState,
    rng: np.random.Generator,
    t: int,
    posts: PostBatch | None,
    engagement_delta: np.ndarray | None,
    exposures: Exposures | None,
    actions: np.ndarray | None,
    camps: np.ndarray | None = None,
) -> None:
    """Mutates `pop.X_stored` (and the derived `pop.X_used`) in place, plus
    `state.Bs`, per spec §2.9's composition. `cfg.dynamics.drift`: "none"
    skips everything, "social" runs channel 2 only, "full" runs both.
    `posts`/`engagement_delta` (channel 1) and `exposures`/`actions`
    (channel 2) all reference the same tick's active-post pool. `camps`
    (C1.3) enables the affect op when the population has the affect block.
    """
    mode = cfg.dynamics.drift
    if mode == "none":
        return

    names = pop.trait_names
    state.ensure_initialized(pop.X_stored)
    k, k_b = block_rates(cfg, names)
    ramp = ramp_factor(t, cfg.dynamics.drift_ramp_ticks)

    n, n_traits = pop.X_stored.shape
    plasticity = pop.X_used[:, names.index("plasticity")]
    stance_cols = [i for i, name in enumerate(names) if name.startswith("stance_")]
    expr_cols = [i for i, name in enumerate(names) if name in EXPRESSION]
    behav_reinforce = [i for i, name in enumerate(names) if name in BEHAVIOR_REINFORCE_COLS]

    gain = np.zeros((n, n_traits))
    state.op_norms = {}

    if mode == "full" and posts is not None and engagement_delta is not None and len(posts) > 0:
        surprise = state.surprise(posts.author, engagement_delta)
        rl = reinforcement_delta(posts, pop, expr, surprise)
        # C3a: expression columns at the channel-1 rate; the behavior columns
        # with a generation path at their own (named) rate
        gain[:, expr_cols] += cfg.dynamics.drift_lr * ramp * rl[:, expr_cols]
        if behav_reinforce:
            gain[:, behav_reinforce] += cfg.dynamics.drift_lr_behavior * ramp * rl[:, behav_reinforce]
        state.op_norms["reinforcement"] = float(np.linalg.norm(
            gain[:, expr_cols + behav_reinforce]) if (expr_cols or behav_reinforce) else 0.0
        )

    ch2_weights = social_weights(cfg)  # C6: repulsion-ablatable table
    if exposures is not None and actions is not None and posts is not None and len(exposures) > 0:
        soc = social_influence_delta(exposures, actions, posts, pop, stance_cols, weights=ch2_weights)
        gain[:, stance_cols] += cfg.dynamics.drift_lr_social * ramp * soc
        state.op_norms["social"] = float(np.linalg.norm(gain[:, stance_cols]))

        # C1.3: the affect op composes here, from the registry, only when the
        # population actually carries the affect block and camps are defined;
        # its weight tables come from config, like channel 2's (C6's pattern)
        if cfg.population.affect and pop.has_affect and camps is not None:
            aff_host, aff_supp = affect_weights(cfg)
            aff = get("drift_op", "affect_update")({
                "exposures": exposures, "actions": actions, "posts": posts,
                "pop": pop, "camps": camps, "lr_affect": cfg.dynamics.lr_affect,
                "hostility_weights": aff_host, "support_weights": aff_supp,
            })
            aff_cols = [i for i, nm in enumerate(names) if nm in ("identification", "animus")]
            gain[:, aff_cols] += ramp * aff[:, aff_cols]
            state.op_norms["affect"] = float(np.linalg.norm(gain[:, aff_cols]))

    noise = rng.normal(0, cfg.dynamics.noise_sigma, size=(n, n_traits))
    # C1.3: the affect block is a STATE with its own named rate (`lr_affect`),
    # not a learned trait — plasticity (a personality trait with population
    # mean ~0.2) would silently shrink the affect channel fivefold and make
    # `lr_affect` mean something different from what the config says. Every
    # other channel keeps the spec §2.9 plasticity gate.
    plasticity_mask = np.ones(n_traits)
    affect_cols = [i for i, nm in enumerate(names) if nm in ("identification", "animus")]
    for i in affect_cols:
        plasticity_mask[i] = 0.0
    gated = plasticity[:, None] * gain * plasticity_mask[None, :]
    pop.X_stored = pop.X_stored + gated + gain * (1.0 - plasticity_mask[None, :]) \
        - k[None, :] * (pop.X_stored - state.Bs) + noise
    state.Bs = state.Bs + k_b[None, :] * (pop.X_stored - state.Bs)

    for i, link in enumerate(pop.links):
        pop.X_used[:, i] = to_used(pop.X_stored[:, i], link)
