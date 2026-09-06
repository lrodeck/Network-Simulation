"""Trait drift (spec §2.9): two free channels plus Ornstein-Uhlenbeck
mean-reversion. Operates on stored (unconstrained) traits, so it is a plain
additive process that can never leave the feasible set (spec §1.1).

Channel 3 (LLM adjudication) is queued only — gated, event-triggered, and
deferred to step 12; `llm_adjudication` in config stays unused here.
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
from discourse_lab.population.traits import EXPRESSION, trait_table

# Per-block reversion rate (spec §2.9): expression reverts fast (style is
# fashion), stance reverts slowly, personality effectively not at all.
DEFAULT_K: dict[str, float] = {
    "personality": 0.0,
    "expression": 0.05,
    "topic_affinity": 0.02,
    "stance": 0.005,
    "behavior": 0.01,
    "meta": 0.01,
}

# Channel 2 action weights (spec §2.9). `quote` is not in the spec's table;
# treated as a weaker rebroadcast than a plain repost since it carries the
# quoter's own (possibly critical) commentary.
ACTION_WEIGHTS: dict[str, float] = {
    "like": 1.0,
    "repost": 1.5,
    "quote": 0.5,
    "reply": -0.5,
    "report": -2.0,
    "skip": 0.0,
}


@dataclass
class DriftState:
    """`Bs`, the slow-moving mean-reversion target, initialised to `X_stored`
    the first time drift runs (spec §2.9's `Bs` starts at the population's
    own baseline, not zero). `engagement_baseline` is a per-user running
    expectation of their own engagement, needed to make "surprise" mean
    anything for an author who only posted once this tick (spec §2.9's
    `E[engagement | author_p]` reads as a standing expectation, not
    something recomputable from a single observation).
    """

    Bs: np.ndarray | None = field(default=None)
    engagement_baseline: np.ndarray | None = field(default=None)
    baseline_ema: float = 0.2

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
    `cfg.dynamics.ou_k`. `k_b ~= k / 10` (spec §2.9).
    """
    specs = trait_table(cfg)
    assert [s.name for s in specs] == trait_names
    overrides = dict(cfg.dynamics.ou_k)

    k = np.array([overrides.get(s.block, DEFAULT_K[s.block]) for s in specs])
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
    """Channel 1 (spec §2.9): a bandit-style pull of each author's expression
    traits toward whatever they posted when it did better than expected.

    `surprise` (`r_p`, from `DriftState.surprise`) weights each post's
    contribution. The post's *predicted* style, `X_stored[author] @ A.T`, is
    the expression map's own trait->post-dims map restricted to expression
    columns; the actual style is recovered by inverting each dim's link on
    the post's used-space value. Moving expression traits by
    `A_expr^T @ residual` is exactly the gradient of that linear map — the
    direction that would have produced more of the engaging residual.

    Averaged (not summed) over an author's posts this tick, so the step size
    is set by `drift_lr` alone rather than incidentally scaling with how
    many times a busy author happened to post.
    """
    n = pop.X_stored.shape[0]
    names = pop.trait_names
    expr_cols = [i for i, name in enumerate(names) if name in EXPRESSION]
    delta = np.zeros((n, len(expr_cols)))
    if len(posts) == 0:
        return delta

    author = posts.author
    actual_stored = np.stack([to_stored(getattr(posts, dim), POST_DIM_LINKS[dim]) for dim in POST_DIMS], axis=1)
    predicted_stored = pop.X_stored[author] @ expr.A.T
    residual = actual_stored - predicted_stored  # (M, |POST_DIMS|)

    a_expr = expr.A[:, expr_cols]  # (|POST_DIMS|, |expr_cols|)
    contribution = surprise[:, None] * (residual @ a_expr)  # (M, |expr_cols|)

    counts = np.zeros(n)
    np.add.at(counts, author, 1.0)
    np.add.at(delta, author, contribution)
    delta[counts > 0] /= counts[counts > 0, None]
    return delta


_WEIGHT_KEYS = np.array(sorted(ACTION_WEIGHTS))
_WEIGHT_VALS = np.array([ACTION_WEIGHTS[k] for k in _WEIGHT_KEYS])


def _action_weights(actions: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(_WEIGHT_KEYS, actions)
    idx = np.clip(idx, 0, len(_WEIGHT_KEYS) - 1)
    hit = _WEIGHT_KEYS[idx] == actions
    return np.where(hit, _WEIGHT_VALS[idx], 0.0)


def social_influence_delta(
    exposures: Exposures,
    actions: np.ndarray,
    posts: PostBatch,
    pop: Population,
    stance_cols: list[int],
) -> np.ndarray:
    """Channel 2 (spec §2.9): exposure-weighted pull of stance toward what a
    user consumed and did not reject. `posts` is whichever batch
    `exposures.post_idx` indexes into.

    Averaged (not summed) over a user's exposures this tick, for the same
    reason as channel 1: the step size should be set by `drift_lr_social`,
    not by how many posts a heavy feed happened to serve this tick.
    """
    n = pop.X_used.shape[0]
    d = len(stance_cols)
    delta = np.zeros((n, d))
    if len(exposures) == 0:
        return delta

    # vectorised lookup: a list comprehension here is a per-exposure Python
    # loop over tens of thousands of actions each tick (spec §0.5)
    weights = _action_weights(actions)
    user_stance = pop.X_used[exposures.user_id][:, stance_cols]
    post_stance = posts.stance[exposures.post_idx]
    contribution = weights[:, None] * (post_stance - user_stance)

    counts = np.zeros(n)
    np.add.at(counts, exposures.user_id, 1.0)
    np.add.at(delta, exposures.user_id, contribution)
    delta[counts > 0] /= counts[counts > 0, None]
    return delta


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
) -> None:
    """Mutates `pop.X_stored` (and the derived `pop.X_used`) in place, plus
    `state.Bs`, per spec §2.9's composition. `cfg.dynamics.drift`: "none"
    skips everything, "social" runs channel 2 only, "full" runs both.
    `posts`/`engagement_delta` (channel 1) and `exposures`/`actions`
    (channel 2) all reference the same tick's active-post pool.
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

    gain = np.zeros((n, n_traits))

    if mode == "full" and posts is not None and engagement_delta is not None and len(posts) > 0:
        surprise = state.surprise(posts.author, engagement_delta)
        rl = reinforcement_delta(posts, pop, expr, surprise)
        gain[:, expr_cols] += cfg.dynamics.drift_lr * ramp * rl

    if exposures is not None and actions is not None and posts is not None and len(exposures) > 0:
        soc = social_influence_delta(exposures, actions, posts, pop, stance_cols)
        gain[:, stance_cols] += cfg.dynamics.drift_lr_social * ramp * soc

    noise = rng.normal(0, cfg.dynamics.noise_sigma, size=(n, n_traits))
    pop.X_stored = pop.X_stored + plasticity[:, None] * gain - k[None, :] * (pop.X_stored - state.Bs) + noise
    state.Bs = state.Bs + k_b[None, :] * (pop.X_stored - state.Bs)

    for i, link in enumerate(pop.links):
        pop.X_used[:, i] = to_used(pop.X_stored[:, i], link)
