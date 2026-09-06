"""The spec §5.1 stylized-fact measurements.

`stylized_facts_report` existed before this but had never been run against a
real simulation — nothing computed the values it checks. These tests cover
the computations, and the last one runs the whole thing end to end on an
actual run.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.measures import gini
from discourse_lab.metrics import STYLIZED_FACT_RANGES, stylized_facts_from_run, stylized_facts_report
from discourse_lab.metrics.powerlaw import MIN_TAIL, ccdf, powerlaw_fit
from discourse_lab.metrics.stylized import (
    cascade_singleton_share,
    cascade_sizes,
    inter_cluster_interaction,
    lorenz_curve,
    posting_volume_gini,
    stance_clusters,
    thread_depth_mean,
    thread_depths,
)
from discourse_lab.network import cached_graph, generate_graph
from discourse_lab.network.measures import clustering_vs_random, configuration_null, reciprocity
from discourse_lab.population import cached_population, sample_population
from discourse_lab.runner import cached_run, load_run, phase_rngs


@pytest.fixture(autouse=True)
def _quiet_cascade_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="cascade:")
        yield


# --------------------------------------------------------------------------
# power law
# --------------------------------------------------------------------------


def test_powerlaw_recovers_a_known_exponent():
    """Discrete Zipf draws with a known alpha. The continuous MLE is biased
    several percent low on small integer counts, which is why the discrete
    likelihood is maximised instead.
    """
    estimates = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        estimates.append(powerlaw_fit(rng.zipf(2.5, size=20_000)).alpha)

    assert abs(float(np.mean(estimates)) - 2.5) < 0.15


def test_powerlaw_orders_steeper_and_shallower_tails_correctly():
    rng = np.random.default_rng(0)
    shallow = powerlaw_fit(rng.zipf(2.0, size=20_000)).alpha
    steep = powerlaw_fit(rng.zipf(3.5, size=20_000)).alpha
    assert shallow < steep


def test_powerlaw_refuses_to_guess_on_a_short_tail():
    """A number that looks authoritative and is not is worse than nan."""
    fit = powerlaw_fit(np.arange(1, MIN_TAIL - 10, dtype=float))
    assert not np.isfinite(fit.alpha)
    assert not fit  # __bool__ reflects it too


def test_ccdf_is_monotone_and_starts_at_one():
    values, survival = ccdf(np.array([1.0, 2, 2, 5, 10]))
    assert survival[0] == 1.0
    assert np.all(np.diff(survival) <= 0)
    assert np.all(np.diff(values) >= 0)


# --------------------------------------------------------------------------
# cascades
# --------------------------------------------------------------------------


def test_cascade_singleton_share_is_one_when_nothing_branched():
    root = np.arange(50)  # every post is its own root
    assert cascade_singleton_share(root) == 1.0
    np.testing.assert_array_equal(cascade_sizes(root), np.ones(50))


def test_cascade_singleton_share_falls_as_cascades_branch():
    # 8 singletons, 2 cascades of 3
    root = np.concatenate([np.arange(8), np.repeat([100, 200], 3)])
    assert cascade_singleton_share(root) == pytest.approx(0.8)
    assert sorted(cascade_sizes(root)) == [1] * 8 + [3, 3]


def test_thread_depth_is_conditioned_on_branched_cascades():
    """The spec's own table asks for >90% singletons AND mean depth 1.5-3,
    which cannot both hold unconditionally — singletons would drag the mean
    to ~0. Depth is therefore measured only over cascades that branched.
    """
    # one singleton at depth 0, one cascade reaching depth 2, one reaching 4
    root = np.array([0, 1, 1, 1, 2, 2])
    depth = np.array([0, 0, 1, 2, 0, 4])

    np.testing.assert_array_equal(sorted(thread_depths(root, depth)), [2, 4])
    assert thread_depth_mean(root, depth) == pytest.approx(3.0)

    # counted unconditionally the singleton drags it down — the failure mode
    assert thread_depth_mean(root, depth, min_size=1) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# inequality
# --------------------------------------------------------------------------


def test_posting_volume_gini_counts_users_who_never_posted():
    """Lurkers are most of the population by construction; dropping them
    understates inequality badly.
    """
    author = np.zeros(100, dtype=np.int64)  # one user made every post

    with_lurkers = posting_volume_gini(author, n_users=1000)
    posters_only = gini(np.array([100.0]))

    assert with_lurkers > 0.9  # 999 users at zero, one at 100
    assert posters_only == 0.0  # the same data, silently reported as perfectly equal


def test_lorenz_curve_spans_the_unit_square():
    population, value = lorenz_curve(np.array([1.0, 1, 1, 1]))
    assert population[0] == 0.0 and population[-1] == 1.0
    assert value[0] == 0.0 and value[-1] == 1.0
    np.testing.assert_allclose(value, population, atol=1e-9)  # equality => diagonal

    _, skewed = lorenz_curve(np.array([0.0, 0, 0, 100]))
    assert skewed[1] < 0.5  # bows away from the diagonal


# --------------------------------------------------------------------------
# cross-cluster contact
# --------------------------------------------------------------------------


def test_stance_clusters_split_at_the_median():
    stance = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = stance_clusters(stance)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_inter_cluster_interaction_rate_and_hostility():
    labels = np.array([0, 0, 1, 1])
    # user 0 -> author 1 (same camp, like); user 0 -> author 2 (cross, reply)
    users = np.array([0, 0, 2])
    authors = np.array([1, 2, 3])
    actions = np.array(["like", "reply", "like"])

    rate, hostility = inter_cluster_interaction(users, authors, actions, labels)
    assert rate == pytest.approx(1 / 3)
    assert hostility == pytest.approx(1.0)  # the one crossing was a reply


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


def _graph(reciprocity_rate: float, seed: int = 0, n_users: int = 800):
    cfg = dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=n_users),
        graph=dataclasses.replace(Config().graph, mirror_p=reciprocity_rate),
    )
    rng = np.random.default_rng(seed)
    pop = sample_population(cfg, rng)
    return generate_graph(cfg, pop, rng)


def test_measured_reciprocity_tracks_the_configured_rate():
    """Only the *generator* existed before; the target range 0.2-0.4 was
    unverifiable because nothing measured the result.
    """
    low = reciprocity(_graph(0.0).csr)
    high = reciprocity(_graph(0.6).csr)
    assert 0.0 <= low < high <= 1.0


def test_configuration_null_preserves_both_degree_sequences_exactly():
    """Double-edge swaps, so this is exact — no edges lost to collapsed
    duplicates, which would depress null clustering and flatter the model.
    """
    graph = _graph(0.2, n_users=400)
    null = configuration_null(graph.csr, np.random.default_rng(0))

    np.testing.assert_array_equal(
        np.asarray(graph.csr.sum(axis=1)).ravel(), np.asarray(null.sum(axis=1)).ravel()
    )
    np.testing.assert_array_equal(
        np.asarray(graph.csr.sum(axis=0)).ravel(), np.asarray(null.sum(axis=0)).ravel()
    )
    assert null.diagonal().sum() == 0  # still simple
    # and it actually rewired rather than handing back the same graph
    assert (null != (graph.csr > 0)).nnz > 0


def test_homophilous_graph_is_more_clustered_than_its_degree_matched_null():
    graph = _graph(0.2, n_users=600)
    observed, null_mean, ratio = clustering_vs_random(graph.csr, np.random.default_rng(0))
    assert observed > null_mean
    assert ratio > 1.0


# --------------------------------------------------------------------------
# the report itself
# --------------------------------------------------------------------------


def test_report_omits_facts_it_was_not_given():
    report = stylized_facts_report(reciprocity=0.3)
    assert set(report) == {"reciprocity"}
    assert report["reciprocity"]["in_range"] is True


def test_report_rejects_a_fact_it_does_not_know():
    with pytest.raises(ValueError, match="unknown stylized fact"):
        stylized_facts_report(made_up_fact=1.0)


def test_report_covers_all_eight_spec_rows():
    assert len(STYLIZED_FACT_RANGES) == 8


def test_stylized_facts_from_a_real_run_are_all_computed(tmp_path, monkeypatch):
    """The end-to-end check: every fact the run has data for comes back
    populated and finite.

    Deliberately NOT asserting `in_range` — whether the model is calibrated
    is a finding to report, not something a test should be able to force by
    loosening a bound.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=600),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=25, drift="none"),
    )

    cached_run(cfg, seed=0, persist=("posts", "engagements"))
    handle = load_run(cfg, seed=0)
    rngs = phase_rngs(0)
    pop = cached_population(cfg, 0, rngs["population"])
    graph = cached_graph(cfg, 0, pop, rngs["graph"])

    report = stylized_facts_from_run(handle, graph=graph, pop=pop)

    assert set(STYLIZED_FACT_RANGES) <= set(report)
    for name, entry in report.items():
        assert np.isfinite(entry["value"]), f"{name} came back non-finite"
        assert entry["label"]


# --------------------------------------------------------------------------
# Hawkes reply scheduling (spec §2.3, §3.1 step 2)
# --------------------------------------------------------------------------


def test_hawkes_intensity_is_self_exciting_and_threads_age_out():
    from discourse_lab.dynamics.hawkes import HawkesThreads

    th = HawkesThreads()
    th.open_thread(1, mu=5.0)          # mu high enough that replies fire reliably
    rng = np.random.default_rng(0)

    th.step(rng, alpha=0.9, beta=1.5, max_age=10)
    assert th.excitation[0] > 0        # the replies it drew excite the thread

    assert len(th) == 1
    for _ in range(12):
        th.step(rng, alpha=0.9, beta=1.5, max_age=10)
    assert len(th) == 0                # aged out past max_age


def test_hawkes_draws_more_replies_on_hotter_threads():
    """Self-excitation is the whole point: a thread that has been replied to
    should attract more replies than an equally-aged cold one.
    """
    from discourse_lab.dynamics.hawkes import HawkesThreads

    hot, cold = HawkesThreads(), HawkesThreads()
    hot.open_thread(1, mu=2.0)
    cold.open_thread(1, mu=2.0)
    hot.excitation[0] = 20.0           # a thread already in full flow

    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    n_hot = sum(hot.step(rng_a, 0.9, 1.5, 50).get(1, 0) for _ in range(5))
    n_cold = sum(cold.step(rng_b, 0.9, 1.5, 50).get(1, 0) for _ in range(5))
    assert n_hot > n_cold


def test_reply_posts_are_generated_not_derived_from_engagement():
    """spec §2.7 keeps cascades to repost/quote; reply *posts* come from the
    Hawkes draw in the generation phase. Both facts, asserted together,
    because conflating them is exactly the mistake this replaced.
    """
    from discourse_lab.dynamics.cascade import BRANCHING_ACTIONS, CASCADE_ACTIONS

    assert CASCADE_ACTIONS == ("repost", "quote")
    assert BRANCHING_ACTIONS == ("repost", "quote")

    import discourse_lab.dynamics.timing as timing
    assert not hasattr(timing, "HawkesThreads"), "the duplicate is back"

    from discourse_lab.dynamics import HawkesThreads as exported
    from discourse_lab.dynamics.hawkes import HawkesThreads as canonical
    assert exported is canonical  # one class, one name


def test_a_run_produces_reply_posts_with_real_thread_depth():
    cfg = dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=600),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=30, drift="none"),
    )
    kinds, depths = set(), []
    for state in __import__("discourse_lab.runner", fromlist=["run_iter"]).run_iter(cfg, seed=0):
        if state.retired_posts is not None and len(state.retired_posts) > 0:
            kinds.update(np.unique(state.retired_posts.kind).tolist())
            depths.extend(state.retired_posts.depth.tolist())

    assert "reply" in kinds, "the Hawkes draw produced no reply posts at all"
    assert max(depths) > 1, "no thread got past depth 1"


# --------------------------------------------------------------------------
# graph consequences of spec §2.2's uniform long ties
# --------------------------------------------------------------------------


def test_uniform_long_ties_cap_the_in_degree_tail():
    """§2.2 mandates a *uniform* long-tie component. The consequence is that
    the population's Pareto prominence tail does not reach the graph: within
    the kNN pool you can only be followed by the users whose neighbourhood
    contains you, and a uniform long tie is as likely to land on a nobody.

    This test pins the limitation in place so it is a known property rather
    than a surprise in a figure. If it ever starts failing because in-degree
    got heavy-tailed, the generator changed and §5.1's engagement rows should
    be re-checked.
    """
    cfg = dataclasses.replace(
        Config(), population=dataclasses.replace(Config().population, n_users=2000)
    )
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    graph = generate_graph(cfg, pop, rng)

    in_degree = np.asarray(graph.csr.sum(axis=0)).ravel().astype(float)
    fit = powerlaw_fit(in_degree)

    # measured alpha ~4.9 at N=2000, ~7.9 at N=3000 — scale-dependent, but far
    # from the 2-3 §5.1's engagement rows need
    assert fit.alpha > 4.0, "in-degree became heavy-tailed; re-check spec §5.1"
    assert in_degree.max() < 10 * in_degree.mean()


def test_knn_pool_must_leave_room_for_the_preference_weights():
    """At knn_k <= mean_degree every candidate is taken, so homophily_beta and
    prominence_gamma stop doing anything at all.
    """
    from discourse_lab.config import GraphConfig

    with pytest.raises(ValueError, match="inert"):
        GraphConfig(knn_k=40, mean_degree=40.0)


def test_attention_concentration_comes_from_the_feed_not_the_graph(tmp_path, monkeypatch):
    """spec §5.1's attention Gini (0.8-0.95) is a claim about engagement-
    optimised platforms, and this pins where the concentration comes from.

    A chronological feed orders by time, so exposure is spread evenly across
    every active post regardless of who wrote it, and no post can run away.
    Measured at n_users=3000, n_ticks=60: the largest engagement count any
    post ever reached was 15 under `chronological`, 32 under `popularity`, and
    733 under `engagement_optimized` + `bandwagon` (Gini 0.609 / 0.706 /
    0.887). Making the graph heavy-tailed instead does not do it — raising max
    in-degree tenfold via the `latent_pa` generator left the Gini at 0.609.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    base = dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=600),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=25, drift="none"),
    )

    def max_engagement(ranker: str, kernel: str) -> int:
        cfg = dataclasses.replace(
            base, dynamics=dataclasses.replace(base.dynamics, ranker=ranker, kernel=kernel)
        )
        cached_run(cfg, seed=0, persist=("posts",))
        return int(load_run(cfg, seed=0).posts()["engagement_count"].max())

    chronological = max_engagement("chronological", "homophily")
    optimized = max_engagement("engagement_optimized", "bandwagon")

    assert optimized > 3 * chronological, (
        f"engagement_optimized+bandwagon peaked at {optimized}, chronological at "
        f"{chronological} — the feed is no longer concentrating attention"
    )
