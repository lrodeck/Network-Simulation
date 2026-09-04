"""Step 5 verification (dev §6): two-peak circadian shape, Hawkes burstiness
and stability, post generation (topic softmax, conformity line, A/B/C maps),
stub renderer format.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from discourse_lab.config import Config
from discourse_lab.dynamics import (
    ExpressionMap,
    FatigueState,
    HawkesThreads,
    circadian_factor,
    circadian_shape,
    generate_posts,
    sample_post_counts,
    stub_render,
)
from discourse_lab.dynamics.posts import sample_stance, sample_topics
from discourse_lab.population import sample_population


def test_circadian_shape_has_two_peaks_and_unit_mean():
    shape = circadian_shape(96)  # 15-min ticks
    assert abs(shape.mean() - 1.0) < 1e-9

    # local maxima: points strictly greater than both neighbours (wrap-around)
    left = np.roll(shape, 1)
    right = np.roll(shape, -1)
    peaks = np.sum((shape > left) & (shape > right))
    assert peaks == 2


def test_fatigue_suppresses_after_a_burst():
    fatigue = FatigueState.initial(3)
    fatigue.step(n_posts=np.array([0, 5, 0]), decay=0.9)
    assert fatigue.factor()[1] < fatigue.factor()[0]
    assert fatigue.factor()[1] < fatigue.factor()[2]


def test_poisson_posting_rate_scales_with_activity_and_circadian_factor():
    rng = np.random.default_rng(0)
    activity = np.full(2000, 2.0)
    shape = circadian_shape(24)
    circ = circadian_factor(shape.argmax(), 24, np.zeros(2000), shape)  # at the peak
    fatigue = np.ones(2000)
    counts = sample_post_counts(rng, activity, circ, fatigue)
    assert abs(counts.mean() - 2.0 * shape.max()) < 0.3


def test_hawkes_replies_are_burstier_than_poisson_with_same_mean_rate():
    """A high alpha/beta ratio (below 1) should cluster replies in time far
    more than an independent Poisson process with the same average rate.
    Averaged over many independent threads (rather than one long run) to keep
    the dispersion estimate itself low-noise.
    """
    rng = np.random.default_rng(1)
    beta = 1.5
    mu0 = 0.05
    alpha = 0.9 * beta  # ratio 0.9, close to the instability boundary
    n_threads = 800
    n_ticks = 40

    threads = HawkesThreads()
    for pid in range(n_threads):
        threads.open_thread(post_id=pid, mu=mu0)

    totals = np.zeros(n_threads)
    for _ in range(n_ticks):
        result = threads.step(rng, alpha, beta, max_age=10_000)
        for pid, n in result.items():
            totals[pid] += n

    poisson_totals = rng.poisson(totals.mean(), size=n_threads)

    # variance-to-mean ratio (index of dispersion): Poisson ~ 1, Hawkes >> 1
    hawkes_dispersion = totals.var() / max(totals.mean(), 1e-9)
    poisson_dispersion = poisson_totals.var() / max(poisson_totals.mean(), 1e-9)
    assert hawkes_dispersion > 1.8 * poisson_dispersion


def test_hawkes_thread_closes_after_max_age():
    rng = np.random.default_rng(2)
    threads = HawkesThreads()
    threads.open_thread(post_id=7, mu=0.1)
    for _ in range(5):
        threads.step(rng, alpha=0.1, beta=1.0, max_age=5)
    assert len(threads) == 0


def test_topic_sampling_follows_softmax_of_affinity_plus_attention():
    rng = np.random.default_rng(3)
    n, k = 20_000, 4
    affinity = np.zeros((n, k))
    affinity[:, 2] = 3.0  # strong prior affinity for topic 2
    s_t = np.zeros(k)

    topics = sample_topics(rng, affinity, s_t, eta=0.5)
    assert (topics == 2).mean() > 0.8  # softmax(3 vs 0,0,0) ~ 0.87

    # now attention strongly favours topic 0; softmax should shift there for eta large
    s_t2 = np.zeros(k)
    s_t2[0] = 20.0
    topics2 = sample_topics(rng, affinity, s_t2, eta=1.0)
    assert (topics2 == 0).mean() > (topics == 0).mean()


def test_conviction_one_disables_conformity():
    rng = np.random.default_rng(4)
    n, d = 5000, 2
    conviction = np.ones(n)
    stance_u = rng.normal(0, 1, size=(n, d))
    topic_p = np.zeros(n, dtype=int)
    sigma_t = np.array([[5.0, -5.0]])  # far from stance_u; would pull hard if conformity were active

    stance_p = sample_stance(rng, conviction, stance_u, topic_p, sigma_t, noise_sigma=0.0)
    np.testing.assert_allclose(stance_p, stance_u)


def test_low_conviction_pulls_toward_dominant_topic_stance():
    rng = np.random.default_rng(5)
    n, d = 5000, 2
    conviction = np.zeros(n)
    stance_u = rng.normal(0, 1, size=(n, d))
    topic_p = np.zeros(n, dtype=int)
    sigma_t = np.array([[5.0, -5.0]])

    stance_p = sample_stance(rng, conviction, stance_u, topic_p, sigma_t, noise_sigma=0.0)
    np.testing.assert_allclose(stance_p, np.broadcast_to(sigma_t[0], stance_p.shape))


def test_expression_map_reproduces_authored_traits():
    cfg = dataclasses.replace(Config(), population=dataclasses.replace(Config().population, n_users=20_000))
    rng = np.random.default_rng(6)
    pop = sample_population(cfg, rng)

    K = cfg.population.n_topics
    expr = ExpressionMap.build(pop.trait_names, K)
    s_t = np.zeros(K)
    topic_p = np.zeros(len(pop.X_used), dtype=int)

    d = expr.generate(pop.X_stored, topic_p, s_t, rng, noise_sigma=0.05)

    neuroticism = pop.X_used[:, pop.trait_names.index("neuroticism")]
    # spec example claim: high neuroticism raises arousal
    corr = np.corrcoef(neuroticism, d["arousal"])[0, 1]
    assert corr > 0.2

    agreeableness = pop.X_used[:, pop.trait_names.index("agreeableness")]
    # spec example claim: low agreeableness raises provocativeness
    corr2 = np.corrcoef(agreeableness, d["provocativeness"])[0, 1]
    assert corr2 < -0.2


def test_generate_posts_end_to_end_and_stub_renderer_format():
    cfg = Config()
    rng = np.random.default_rng(7)
    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()

    expr = ExpressionMap.build(pop.trait_names, K)
    authors = rng.choice(cfg.population.n_users, size=30, replace=True)
    posts = generate_posts(authors, pop, expr, np.zeros(K), np.zeros((K, D)), eta=0.3, rng=rng)

    assert len(posts) == 30
    for name in ("arousal", "provocativeness", "novelty", "specificity", "quality"):
        vals = getattr(posts, name)
        assert (vals >= 0).all() and (vals <= 1).all()
    assert (posts.valence >= -1).all() and (posts.valence <= 1).all()

    lines = stub_render(posts)
    assert len(lines) == 30
    for author, topic, line in zip(posts.author, posts.topic, lines):
        assert line.startswith(f"[u{author} - topic{topic} - ")
        assert line.endswith("]")
