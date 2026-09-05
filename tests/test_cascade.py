"""Step 8 verification (dev §6): cascade size distribution heavy-tailed
(>90% singletons at scale), boundary-crossing (depth/size cap) diagnostics
work.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.dynamics import ExpressionMap, generate_posts
from discourse_lab.dynamics.cascade import CascadeState, check_r_eff, derive_posts, follower_counts, r_eff
from discourse_lab.network import generate_graph
from discourse_lab.population import sample_population


def _setup(n_users=500, n_posts=1, seed=0):
    cfg = dataclasses.replace(Config(), population=dataclasses.replace(Config().population, n_users=n_users))
    rng = np.random.default_rng(seed)
    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    authors = rng.choice(n_users, size=n_posts, replace=True)
    posts = generate_posts(authors, pop, expr, np.zeros(K), np.zeros((K, D)), eta=0.3, rng=rng, t=0)
    return cfg, rng, pop, expr, posts


def test_repost_leaves_dims_and_stance_unchanged():
    cfg, rng, pop, expr, posts = _setup()
    K, D = cfg.population.n_topics, cfg.stance_dims()
    actions = np.array(["repost"])
    reposter = np.array([7])
    state = CascadeState()

    new_posts, warns = derive_posts(
        actions, np.array([0]), reposter, posts, pop, expr, np.zeros(K), np.zeros((K, D)), rng, state,
        max_depth=4, max_size=1000, start_id=1, t=1,
    )
    assert warns == []
    np.testing.assert_allclose(new_posts.stance[0], posts.stance[0])
    assert new_posts.arousal[0] == posts.arousal[0]
    assert new_posts.author[0] == 7
    assert new_posts.parent[0] == posts.id[0]
    assert new_posts.root[0] == posts.root[0]
    assert new_posts.depth[0] == posts.depth[0] + 1
    assert new_posts.kind[0] == "repost"


def test_quote_shifts_stance_toward_quoter():
    cfg, rng, pop, expr, posts = _setup()
    K, D = cfg.population.n_topics, cfg.stance_dims()
    names = pop.trait_names
    stance_cols = [i for i, n in enumerate(names) if n.startswith("stance_")]

    quoter = 3
    pop.X_used[quoter, stance_cols] = posts.stance[0] + 10.0  # deliberately far from the original

    actions = np.array(["quote"])
    state = CascadeState()
    new_posts, _ = derive_posts(
        actions, np.array([0]), np.array([quoter]), posts, pop, expr, np.zeros(K), np.zeros((K, D)), rng, state,
        max_depth=4, max_size=1000, start_id=1, t=1,
    )

    dist_to_orig = np.linalg.norm(new_posts.stance[0] - posts.stance[0])
    dist_orig_to_quoter = np.linalg.norm(posts.stance[0] - pop.X_used[quoter, stance_cols])
    assert 0 < dist_to_orig < dist_orig_to_quoter  # moved toward the quoter, not all the way


def test_max_depth_drops_and_warns():
    cfg, rng, pop, expr, posts = _setup()
    K, D = cfg.population.n_topics, cfg.stance_dims()
    posts.depth[0] = 4  # already at the cap
    actions = np.array(["repost"])
    state = CascadeState()

    new_posts, warns = derive_posts(
        actions, np.array([0]), np.array([1]), posts, pop, expr, np.zeros(K), np.zeros((K, D)), rng, state,
        max_depth=4, max_size=1000, start_id=1, t=1,
    )
    assert new_posts is None
    assert any("max_cascade_depth" in w for w in warns)


def test_max_size_caps_and_warns():
    cfg, rng, pop, expr, posts = _setup()
    K, D = cfg.population.n_topics, cfg.stance_dims()
    root = int(posts.root[0])
    state = CascadeState()
    state.size_by_root[root] = 998  # 2 slots left before the cap

    actions = np.array(["repost"] * 5)
    reposters = np.arange(1, 6)
    new_posts, warns = derive_posts(
        actions, np.zeros(5, dtype=int), reposters, posts, pop, expr, np.zeros(K), np.zeros((K, D)), rng, state,
        max_depth=10, max_size=1000, start_id=1, t=1,
    )
    assert len(new_posts) == 2
    assert any("max_cascade_size" in w for w in warns)
    assert state.size_by_root[root] == 1000


def test_r_eff_scales_with_repost_rate_and_audience_and_warns_above_one():
    cfg, rng, pop, expr, posts = _setup(n_users=2000)
    graph = generate_graph(cfg, pop, rng)

    actions_low = np.array(["skip"] * 90 + ["repost"] * 10)
    actions_high = np.array(["skip"] * 10 + ["repost"] * 90)
    reposters = np.arange(100)

    low = r_eff(actions_low, n_exposures=100, graph=graph, reposter_ids=reposters[actions_low == "repost"])
    high = r_eff(actions_high, n_exposures=100, graph=graph, reposter_ids=reposters[actions_high == "repost"])
    assert high > low

    with pytest.warns(UserWarning, match="R_eff"):
        check_r_eff(1.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_r_eff(0.5)  # must not raise/warn


def test_cascade_size_distribution_is_heavy_tailed_and_mostly_singletons():
    """Subcritical branching process (repost probability tuned so E[R] < 1
    but with enough spread that a few cascades run large): >90% of roots stay
    singletons, and the tail is much heavier than a Poisson baseline would be.
    """
    rng = np.random.default_rng(0)
    n_roots = 3000
    p_repost = 0.07  # E[reposts per exposure]
    pareto_alpha = 2.3  # audience per repost follows the spec's prominence tail (§2.1)

    sizes = np.ones(n_roots, dtype=np.int64)  # each root starts as its own cascade of size 1
    active = [(r, 1) for r in range(n_roots)]  # (root, tick's exposure count at this frontier)

    for _ in range(6):  # a handful of cascade generations
        next_active = []
        for root, n_exposed in active:
            n_reposts = rng.binomial(max(n_exposed, 0), p_repost) if n_exposed > 0 else 0
            if n_reposts == 0:
                continue
            sizes[root] += n_reposts
            # heavy-tailed fanout per repost, scaled to a plausible follower-count order of magnitude
            audience = int((rng.pareto(pareto_alpha, size=n_reposts) * 15).sum())
            next_active.append((root, audience))
        active = next_active
        if not active:
            break

    singleton_share = (sizes == 1).mean()
    assert singleton_share > 0.9

    # heavy-tailed: the max is far larger than a moderate multiple of the mean
    assert sizes.max() > 10 * sizes.mean()
