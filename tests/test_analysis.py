"""spec §4.2 component protocols, §4.4 grid sweep, §5.4 sensitivity."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from discourse_lab.analysis import (
    MIN_SEEDS,
    SENSITIVITY_PARAMS,
    expand_grid,
    get_param,
    sensitivity,
    set_param,
    sweep,
)
from discourse_lab.config import Config


def _small():
    return dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=200),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=4, drift="none"),
    )


def test_component_protocols_match_the_real_components():
    """spec §4.2: every swappable component is a Protocol. Checked against the
    shipped implementations, so the contract cannot drift from the call site.
    """
    from discourse_lab.dynamics.drift import apply_drift
    from discourse_lab.exposure.kernel import compute_features
    from discourse_lab.network.barabasi import barabasi_albert_graph
    from discourse_lab.network.latent_space import latent_space_graph
    from discourse_lab.network.sbm import sbm_graph
    from discourse_lab.protocols import DriftModel, FeatureMap, GraphGenerator

    for gen in (latent_space_graph, sbm_graph, barabasi_albert_graph):
        assert isinstance(gen, GraphGenerator), gen
    assert isinstance(compute_features, FeatureMap)
    assert isinstance(apply_drift, DriftModel)


def test_set_and_get_param_round_trip_on_dotted_paths():
    cfg = _small()
    changed = set_param(cfg, "dynamics.inject_k", 7)
    assert get_param(changed, "dynamics.inject_k") == 7
    assert get_param(cfg, "dynamics.inject_k") == 0      # base untouched (frozen)
    assert changed.hash() != cfg.hash()


def test_expand_grid_is_the_cartesian_product():
    cfg = _small()
    cells = expand_grid(cfg, {"dynamics.kernel": ["homophily", "null"], "dynamics.inject_k": [0, 3]})
    assert len(cells) == 4
    assert {c["dynamics.kernel"] for c, _ in cells} == {"homophily", "null"}
    assert len({cfg_.hash() for _, cfg_ in cells}) == 4


def test_sweep_returns_tidy_long_form(tmp_path, monkeypatch):
    """spec §4.4: "one row per (config, seed, metric)"."""
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    frame = sweep(
        _small(), {"dynamics.kernel": ["homophily", "null"]}, seeds=[0, 1],
        metrics=["attention_gini", "r_eff"], warn_on_few_seeds=False,
    )
    assert set(frame.columns) == {"dynamics.kernel", "seed", "metric", "value"}
    assert len(frame) == 2 * 2 * 2                     # cells x seeds x metrics
    assert set(frame["metric"].unique()) == {"attention_gini", "r_eff"}


def test_sweep_warns_below_the_minimum_seed_count(tmp_path, monkeypatch):
    """spec §4.4 calls too few seeds "the single most common way simulation
    studies of this kind go wrong", so it is a warning, not a silent default.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    with pytest.warns(UserWarning, match=f"at least {MIN_SEEDS}"):
        sweep(_small(), {}, seeds=[0], metrics=["r_eff"])


def test_sensitivity_covers_the_parameters_the_spec_names(tmp_path, monkeypatch):
    """spec §5.4: one-at-a-time over attention_budget, inject_k,
    hawkes_alpha_beta, ou_k, mean degree.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = sensitivity(
            _small(), seeds=[0], params=("dynamics.attention_budget", "dynamics.inject_k"),
            factors=(0.5, 1.0), metrics=["r_eff"],
        )
    assert set(frame["param"].unique()) == {"dynamics.attention_budget", "dynamics.inject_k"}
    assert "inverts" in frame.columns
    assert frame["param_value"].n_unique() >= 2

    # the spec's five named knobs are all reachable as dotted paths
    cfg = _small()
    for path in SENSITIVITY_PARAMS:
        assert get_param(cfg, path) is not None, path


def test_agreement_metric_removes_the_dimension_dependent_offset():
    """spec §7.5's open choice, as a dial. `euclidean` is the raw stance
    distance, whose mean grows like sqrt(D) — measured -1.13 at D=1, -2.26 at
    D=3, -3.01 at D=5 — while its spread barely moves. Raising D under that
    metric silently subtracts a constant from every utility, which is an
    intercept shift wearing a feature's clothes. `rms` divides by sqrt(D) so
    the feature has the same location at any D and a theta transfers.
    """
    import numpy as np

    from discourse_lab.dynamics import ExpressionMap, generate_posts
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.exposure.kernel import compute_features
    from discourse_lab.population import sample_population

    means = {}
    for metric in ("euclidean", "rms"):
        for D in (1, 4):
            cfg = dataclasses.replace(
                Config(),
                population=dataclasses.replace(Config().population, n_users=500, stance_dims=D),
            )
            rng = np.random.default_rng(0)
            pop = sample_population(cfg, rng)
            K = cfg.population.n_topics
            expr = ExpressionMap.build(pop.trait_names, K)
            authors = rng.choice(500, size=300)
            posts = generate_posts(
                authors, pop, expr, np.zeros(K), np.zeros((K, D)), 0.3, rng, t=0
            )
            m = len(posts)
            ex = Exposures(
                post_idx=np.arange(m), user_id=rng.choice(500, size=m), rank=np.zeros(m, dtype=int)
            )
            feats = compute_features(
                ex, posts, pop, np.ones(m, dtype=bool), 0, agreement_metric=metric
            )
            means[(metric, D)] = float(feats["agreement"].mean())

    euclid_drift = abs(means[("euclidean", 4)] / means[("euclidean", 1)])
    rms_drift = abs(means[("rms", 4)] / means[("rms", 1)])
    assert euclid_drift > 1.4, f"euclidean should grow with D, got ratio {euclid_drift:.2f}"
    assert rms_drift < 1.25, f"rms should be near scale-free in D, got ratio {rms_drift:.2f}"
