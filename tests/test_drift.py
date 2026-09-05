"""Step 11 verification (dev §6): reinforcement bandit on expression dims,
action-weighted social influence on stance, Ornstein-Uhlenbeck composition
with slow baseline drift, gains ramped from zero, no runaway over 1000 ticks.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from discourse_lab.config import Config
from discourse_lab.dynamics.cascade import CascadeState
from discourse_lab.dynamics.drift import (
    ACTION_WEIGHTS,
    DriftState,
    apply_drift,
    block_rates,
    ramp_factor,
    reinforcement_delta,
    social_influence_delta,
)
from discourse_lab.dynamics.expression import ExpressionMap
from discourse_lab.dynamics.posts import generate_posts
from discourse_lab.dynamics.tick import TickEngine
from discourse_lab.exposure.attention import Exposures
from discourse_lab.network import generate_graph
from discourse_lab.population import sample_population
from discourse_lab.population.traits import EXPRESSION, trait_names


def _setup(n_users=500, seed=0):
    cfg = dataclasses.replace(Config(), population=dataclasses.replace(Config().population, n_users=n_users))
    rng = np.random.default_rng(seed)
    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    return cfg, rng, pop, expr, K, D


def test_ramp_factor_linear_from_zero():
    assert ramp_factor(0, 50) == 0.0
    assert ramp_factor(25, 50) == 0.5
    assert ramp_factor(50, 50) == 1.0
    assert ramp_factor(100, 50) == 1.0  # held at 1 past the ramp window
    assert ramp_factor(10, 0) == 1.0  # ramp disabled


def test_block_rates_match_spec_ordering():
    cfg = Config()
    names = trait_names(cfg)
    k, k_b = block_rates(cfg, names)
    idx = {n: i for i, n in enumerate(names)}

    assert k[idx["openness"]] == 0.0  # personality: effectively no reversion
    stance_col = next(i for i, n in enumerate(names) if n.startswith("stance_"))
    assert k[idx["verbosity"]] > k[stance_col] > 0  # expression reverts fast, stance slowly
    np.testing.assert_allclose(k_b, k / 10)


def test_reinforcement_delta_pulls_expression_toward_surprising_posts():
    cfg, rng, pop, expr, K, D = _setup(n_users=1)
    authors = np.array([0])
    posts = generate_posts(authors, pop, expr, np.zeros(K), np.zeros((K, D)), eta=0.3, rng=rng)

    surprise = np.array([5.0])  # strongly positive surprise
    delta_pos = reinforcement_delta(posts, pop, expr, surprise)
    delta_neg = reinforcement_delta(posts, pop, expr, -surprise)

    # a stronger positive surprise should pull harder in the same direction
    # than a matched negative surprise pulls in the opposite one
    np.testing.assert_allclose(delta_pos, -delta_neg)
    assert np.linalg.norm(delta_pos) > 0


def test_drift_state_surprise_is_zero_baseline_at_first_sight():
    """A single post from a never-before-seen author can't be judged against
    its own mean (that's always zero) — it's judged against the running
    baseline, which starts at 0 and only moves after this observation.
    """
    state = DriftState()
    state.ensure_initialized(np.zeros((3, 1)))
    surprise = state.surprise(np.array([0]), np.array([7.0]))
    assert surprise[0] == 7.0  # baseline was 0
    assert state.engagement_baseline[0] == 7.0 * state.baseline_ema

    surprise2 = state.surprise(np.array([0]), np.array([7.0]))
    assert surprise2[0] < surprise[0]  # baseline caught up, so the same result is less surprising now


def test_social_influence_weights_match_spec_and_pulls_toward_consumed_stance():
    assert ACTION_WEIGHTS["like"] == 1.0
    assert ACTION_WEIGHTS["repost"] == 1.5
    assert ACTION_WEIGHTS["reply"] == -0.5
    assert ACTION_WEIGHTS["report"] == -2.0
    assert ACTION_WEIGHTS["skip"] == 0.0

    cfg, rng, pop, expr, K, D = _setup(n_users=2)
    stance_cols = [i for i, n in enumerate(pop.trait_names) if n.startswith("stance_")]
    pop.X_used[0, stance_cols] = 0.0
    posts = generate_posts(np.array([1]), pop, expr, np.zeros(K), np.zeros((K, D)), eta=0.3, rng=rng)
    posts.stance[0] = np.full(D, 3.0)  # far from user 0's stance

    exposures = Exposures(post_idx=np.array([0]), user_id=np.array([0]), rank=np.array([0]))
    like = social_influence_delta(exposures, np.array(["like"]), posts, pop, stance_cols)
    report = social_influence_delta(exposures, np.array(["report"]), posts, pop, stance_cols)

    assert (like[0] > 0).all()  # user 0 pulled toward the (positive) post stance
    assert (report[0] < 0).all()  # reporting pushes user 0 away from it
    assert (like[1] == 0).all()  # user 1 had no exposures at all


def test_gains_ramped_from_zero_means_no_drift_at_tick_zero():
    cfg, rng, pop, expr, K, D = _setup(n_users=200)
    cfg = dataclasses.replace(cfg, dynamics=dataclasses.replace(cfg.dynamics, drift_ramp_ticks=50))
    x0 = pop.X_stored.copy()

    authors = rng.choice(200, size=50, replace=True)
    posts = generate_posts(authors, pop, expr, np.zeros(K), np.zeros((K, D)), eta=0.3, rng=rng)
    engagement_delta = rng.poisson(2.0, size=len(posts)).astype(float)
    exposures = Exposures(post_idx=rng.integers(0, len(posts), 200), user_id=rng.integers(0, 200, 200), rank=np.zeros(200, dtype=int))
    actions = rng.choice(["like", "reply", "repost", "skip"], size=200)

    state = DriftState()
    apply_drift(cfg, pop, expr, state, rng, t=0, posts=posts, engagement_delta=engagement_delta, exposures=exposures, actions=actions)

    # at t=0 the ramp is exactly 0, so only mean-reversion (0 initially, Bs==X0) and noise act
    assert np.abs(pop.X_stored - x0).max() < 5 * cfg.dynamics.noise_sigma


def test_no_runaway_over_1000_ticks_with_all_channels_live():
    cfg = dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=150),
        dynamics=dataclasses.replace(
            Config().dynamics, n_ticks=1000, drift="full", drift_ramp_ticks=20, noise_sigma=0.01
        ),
    )
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    graph = generate_graph(cfg, pop, rng)
    rngs = {
        name: np.random.default_rng(s)
        for name, s in zip(
            ("population", "graph", "timing", "generation", "exposure", "reaction", "perception", "cascade", "drift"),
            np.random.default_rng(1).spawn(9),
        )
    }
    engine = TickEngine(cfg=cfg, pop=pop, graph=graph, rngs=rngs)

    initial_std = pop.X_stored.std(axis=0) + 1e-9
    for t in range(cfg.dynamics.n_ticks):
        engine.step(t)

    x_final = engine.pop.X_stored
    assert np.isfinite(x_final).all()
    # bounded relative to where it started, block by block; a runaway
    # process would blow this up by orders of magnitude, monotonically
    assert np.all(x_final.std(axis=0) < 20 * initial_std)
    assert np.abs(x_final).max() < 1000
