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
# reply self-excitation and preferential long ties (calibration)
# --------------------------------------------------------------------------


def test_hawkes_threads_decay_and_close():
    from discourse_lab.dynamics.timing import HawkesThreads

    th = HawkesThreads.empty()
    th.record(np.array([1]), np.array([2]), t=0, ratio=0.6, beta=1.5)
    opened = th.intensity(np.array([1]), mu0=0.004)[0]

    th.step(t=1, beta=1.5, max_thread_age=10)
    assert th.intensity(np.array([1]), mu0=0.004)[0] < opened  # decayed
    assert th.intensity(np.array([99]), mu0=0.004)[0] == pytest.approx(0.004)  # cold post

    th.step(t=99, beta=1.5, max_thread_age=10)
    assert th.intensity(np.array([1]), mu0=0.004)[0] == pytest.approx(0.004)  # aged out


def test_a_reply_inherits_its_parents_heat_so_chains_can_deepen():
    """The child seeding is what carries activity *down* a chain — without it
    every reply starts cold and depth cannot grow past 1.
    """
    from discourse_lab.dynamics.timing import HawkesThreads

    th = HawkesThreads.empty()
    for _ in range(5):  # a parent that has been replied to repeatedly
        th.record(np.array([1]), np.array([2]), t=0, ratio=0.6, beta=1.5)

    assert th.intensity(np.array([2]), mu0=0.004)[0] > 0.004  # child is born warm
    assert th.excess[2] < th.excess[1]  # but cooler than its parent, so it terminates


def test_reply_actions_now_spawn_posts_at_greater_depth():
    from discourse_lab.dynamics.cascade import BRANCHING_ACTIONS, CASCADE_ACTIONS, STANCE_SHIFT

    assert "reply" in CASCADE_ACTIONS  # it did not, and thread depth was unreachable
    assert "reply" not in BRANCHING_ACTIONS  # but R in spec §2.7 is over reposts
    assert STANCE_SHIFT["repost"] == 0.0 < STANCE_SHIFT["quote"] < STANCE_SHIFT["reply"]


def test_long_ties_are_prominence_weighted_so_hubs_can_form():
    """`prominence` enters as Pareto(2.3); a uniform long-tie draw threw that
    tail away and in-degree came out at alpha ~7.9.
    """
    cfg = dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=2000),
    )
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    graph = generate_graph(cfg, pop, rng)

    in_degree = np.asarray(graph.csr.sum(axis=0)).ravel()
    prominence = pop.X_used[:, pop.trait_names.index("prominence")]

    # the most prominent users are genuinely the most followed. Bounds are
    # loose because prominence is Pareto: only the very top of the tail pulls
    # far away from the mean, and where that top lands is seed-dependent
    # (measured over 3 seeds: top-10 ratio 2.6-2.9, max ratio 4.0-9.4).
    top_prominent = np.argsort(prominence)[-10:]
    assert in_degree[top_prominent].mean() > 2 * in_degree.mean()
    # and a hub exists at all, rather than everyone sitting near mean degree
    assert in_degree.max() > 3.5 * in_degree.mean()
    assert np.corrcoef(np.log1p(prominence), in_degree)[0, 1] > 0.15


def test_knn_pool_must_leave_room_for_the_preference_weights():
    """At knn_k <= mean_degree every candidate is taken, so homophily_beta and
    prominence_gamma stop doing anything at all.
    """
    from discourse_lab.config import GraphConfig

    with pytest.raises(ValueError, match="inert"):
        GraphConfig(knn_k=40, mean_degree=40.0)
