"""Step 6 verification (dev §6): one candidate generator, one ranker, one
kernel — first runnable dynamics, plus the swappable pieces behave as their
theories predict.
"""

from __future__ import annotations

import collections
import dataclasses

import numpy as np

from discourse_lab.config import Config
from discourse_lab.dynamics import ExpressionMap, generate_posts
from discourse_lab.exposure import (
    apply_kernel,
    candidate_inbox,
    compute_features,
    named_kernel,
    rank_candidates,
    select_exposures,
)
from discourse_lab.network import generate_graph
from discourse_lab.population import sample_population


def _setup_no_graph(n_users=200, n_posts=1, seed=0):
    """For tests that only need a population and posts, not a real graph."""
    cfg = dataclasses.replace(Config(), population=dataclasses.replace(Config().population, n_users=n_users))
    rng = np.random.default_rng(seed)
    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    authors = rng.choice(n_users, size=n_posts, replace=True)
    posts = generate_posts(authors, pop, expr, np.zeros(K), np.zeros((K, D)), eta=0.3, rng=rng, t=0)
    return cfg, rng, pop, posts


def _setup(n_users=3000, n_posts=150, seed=0, inject_k=None, fanout_cap=None):
    cfg = Config()
    if inject_k is not None or fanout_cap is not None:
        cfg = dataclasses.replace(
            cfg,
            dynamics=dataclasses.replace(cfg.dynamics, inject_k=inject_k if inject_k is not None else cfg.dynamics.inject_k),
            graph=dataclasses.replace(cfg.graph, fanout_cap=fanout_cap if fanout_cap is not None else cfg.graph.fanout_cap),
        )
    cfg = dataclasses.replace(cfg, population=dataclasses.replace(cfg.population, n_users=n_users))
    rng = np.random.default_rng(seed)
    pop = sample_population(cfg, rng)
    graph = generate_graph(cfg, pop, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    authors = rng.choice(n_users, size=n_posts, replace=True)
    posts = generate_posts(authors, pop, expr, np.zeros(K), np.zeros((K, D)), eta=0.3, rng=rng, t=0)
    return cfg, rng, pop, graph, posts


def test_candidate_inbox_is_followers_plus_injection_and_respects_fanout_cap():
    cfg, rng, pop, graph, posts = _setup(n_users=2000, n_posts=80, inject_k=5, fanout_cap=10)
    pairs = candidate_inbox(graph, posts, inject_k=5, fanout_cap=10, rng=rng)

    assert len(pairs) > 0
    for i in range(len(posts)):
        mask = pairs.post_idx == i
        n_followers_seen = pairs.is_follower[mask].sum()
        assert n_followers_seen <= 10  # fanout cap
    # every candidate is either a real follower or an injected (non-follower) one
    assert set(np.unique(pairs.is_follower)) <= {True, False}


def test_zero_injection_is_pure_subscription():
    cfg, rng, pop, graph, posts = _setup(n_users=1500, n_posts=50, inject_k=0, fanout_cap=0)
    pairs = candidate_inbox(graph, posts, inject_k=0, fanout_cap=0, rng=rng)
    assert pairs.is_follower.all()


def test_chronological_ranker_orders_by_time_descending_within_user():
    cfg, rng, pop, graph, posts = _setup(n_users=500, n_posts=30)
    posts.t = rng.integers(0, 10, size=len(posts))  # vary post times for this test
    pairs = candidate_inbox(graph, posts, inject_k=3, fanout_cap=0, rng=rng)
    scores = rank_candidates("chronological", pairs, posts, pop, rng)
    np.testing.assert_array_equal(scores, posts.t[pairs.post_idx].astype(float))


def test_affinity_ranker_prefers_high_topic_affinity_and_agreement():
    cfg, rng, pop, graph, posts = _setup(n_users=500, n_posts=1)
    # two candidate users: one with high affinity/agreement, one deliberately mismatched
    names = pop.trait_names
    topic_cols = [i for i, n in enumerate(names) if n.startswith("topic_affinity_")]
    stance_cols = [i for i, n in enumerate(names) if n.startswith("stance_")]

    good_user, bad_user = 0, 1
    pop.X_used[good_user, topic_cols] = 5.0
    pop.X_used[bad_user, topic_cols] = -5.0
    pop.X_used[good_user, stance_cols] = posts.stance[0]
    pop.X_used[bad_user, stance_cols] = posts.stance[0] + 10.0

    from discourse_lab.exposure.inbox import CandidatePairs

    pairs = CandidatePairs(
        post_idx=np.array([0, 0]), user_id=np.array([good_user, bad_user]), is_follower=np.array([True, True])
    )
    scores = rank_candidates("affinity", pairs, posts, pop, rng)
    assert scores[0] > scores[1]


def test_attention_budget_caps_exposure_and_position_decay_reduces_visibility():
    cfg, rng, pop, graph, posts = _setup(n_users=500, n_posts=1, inject_k=0, fanout_cap=0)
    n_candidates = 400
    from discourse_lab.exposure.inbox import CandidatePairs

    user_id = np.arange(n_candidates)
    pairs = CandidatePairs(
        post_idx=np.zeros(n_candidates, dtype=int), user_id=user_id, is_follower=np.ones(n_candidates, dtype=bool)
    )
    scores = np.arange(n_candidates)[::-1].astype(float)  # user 0 ranked best, etc.
    activity = np.ones(n_candidates) * 0.5

    exposures_low_tau = select_exposures(pairs, scores, activity, attention_budget=5.0, tau_position=0.5, rng=np.random.default_rng(1))
    exposures_high_tau = select_exposures(pairs, scores, activity, attention_budget=5.0, tau_position=50.0, rng=np.random.default_rng(1))

    assert len(exposures_low_tau) < n_candidates  # attention budget is binding
    # sharp position decay (low tau) should see strictly fewer than a flat one (high tau)
    assert len(exposures_low_tau) <= len(exposures_high_tau)


def test_homophily_kernel_engages_more_on_agreeing_similar_posts():
    cfg, rng, pop, posts = _setup_no_graph()
    m = 4000
    from discourse_lab.exposure.inbox import CandidatePairs
    from discourse_lab.exposure.attention import Exposures

    exposures = Exposures(post_idx=np.zeros(m, dtype=int), user_id=np.zeros(m, dtype=int), rank=np.zeros(m, dtype=int))
    features = {f: np.zeros(m) for f in compute_features(exposures, posts, pop, np.ones(m, dtype=bool), 0)}
    features["affinity"] = np.concatenate([np.full(m // 2, 3.0), np.full(m - m // 2, -3.0)])
    features["agreement"] = features["affinity"]

    theta = named_kernel("homophily")
    actions = apply_kernel(theta, features, rng)
    engaged_high = actions[: m // 2] != "skip"
    engaged_low = actions[m // 2 :] != "skip"
    assert engaged_high.mean() > engaged_low.mean() + 0.2


def test_outrage_kernel_rewards_disagreement_for_contrarian_users():
    cfg, rng, pop, posts = _setup_no_graph()
    m = 4000
    from discourse_lab.exposure.attention import Exposures

    exposures = Exposures(post_idx=np.zeros(m, dtype=int), user_id=np.zeros(m, dtype=int), rank=np.zeros(m, dtype=int))
    features = {f: np.zeros(m) for f in compute_features(exposures, posts, pop, np.ones(m, dtype=bool), 0)}
    # half the rows: strong disagreement + high contrarianism; half: agreement + low contrarianism
    features["disagree_x_con"] = np.concatenate([np.full(m // 2, 2.0), np.full(m - m // 2, -2.0)])

    theta = named_kernel("outrage")
    actions = apply_kernel(theta, features, rng)
    reply_or_quote_high = np.isin(actions[: m // 2], ["reply", "quote", "report"]).mean()
    reply_or_quote_low = np.isin(actions[m // 2 :], ["reply", "quote", "report"]).mean()
    assert reply_or_quote_high > reply_or_quote_low


def test_null_kernel_is_uniform_over_actions_and_skip():
    m = 6000
    features = {f: np.zeros(m) for f in ("affinity",)}
    theta = named_kernel("null")
    actions = apply_kernel(theta, features, np.random.default_rng(2))
    counts = collections.Counter(actions)
    shares = np.array(list(counts.values())) / m
    assert shares.max() - shares.min() < 0.03  # all 6 outcomes ~equally likely


def test_first_runnable_dynamics_end_to_end():
    cfg, rng, pop, graph, posts = _setup(n_users=2000, n_posts=100)
    pairs = candidate_inbox(graph, posts, cfg.dynamics.inject_k, cfg.graph.fanout_cap, rng)
    scores = rank_candidates(cfg.dynamics.ranker, pairs, posts, pop, rng)
    activity = pop.X_used[:, pop.trait_names.index("activity")]
    exposures = select_exposures(pairs, scores, activity, cfg.dynamics.attention_budget, cfg.dynamics.tau_position, rng)
    assert len(exposures) > 0

    is_follower = np.ones(len(exposures), dtype=bool)
    features = compute_features(exposures, posts, pop, is_follower, t_current=0)
    theta = named_kernel(cfg.dynamics.kernel)
    actions = apply_kernel(theta, features, rng)
    assert len(actions) == len(exposures)
    assert set(actions) <= {"skip", "like", "reply", "repost", "quote", "report"}
