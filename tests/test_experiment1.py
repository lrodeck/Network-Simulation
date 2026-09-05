"""Step 10 verification (dev §6): sweep kernels/rankers over multiple seeds
(min 10) via the flat seed-major queue; every effect measured against its
matched kernel="null" run.
"""

from __future__ import annotations

import dataclasses

from discourse_lab.config import Config
from discourse_lab.experiments import (
    DEFAULT_KERNELS,
    DEFAULT_RANKERS,
    build_experiment1,
    run_experiment1,
    summarize_experiment1,
)


def _base_cfg(n_users=300, n_ticks=6):
    return dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=n_users),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=n_ticks),
    )


def test_default_kernels_and_rankers():
    assert "null" not in DEFAULT_KERNELS
    assert {"homophily", "outrage", "bandwagon", "epistemic"} <= set(DEFAULT_KERNELS)
    assert {"chronological", "random", "popularity", "affinity", "engagement_optimized"} <= set(DEFAULT_RANKERS)


def test_sweep_runs_at_least_ten_seeds_per_cell(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    base = _base_cfg()
    cells = build_experiment1(base, kernels=("homophily",), rankers=("chronological",))
    assert len(cells) == 1

    seeds = list(range(10))
    rows = run_experiment1(cells, seeds)
    assert len(rows) == 10
    assert {r["seed"] for r in rows} == set(seeds)


def test_sweep_is_resumable_by_construction(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    base = _base_cfg(n_users=150, n_ticks=3)
    cells = build_experiment1(base, kernels=("bandwagon",), rankers=("popularity",))

    rows1 = run_experiment1(cells, seeds=[0, 1])
    from discourse_lab.runner import run_dir

    complete_marker = run_dir(cells[0].cfg, 0) / "COMPLETE"
    mtime1 = complete_marker.stat().st_mtime_ns

    rows2 = run_experiment1(cells, seeds=[0, 1, 2])  # re-run overlapping seeds + one new
    assert complete_marker.stat().st_mtime_ns == mtime1  # seed 0 was not recomputed
    assert len(rows2) == 3


def test_null_comparison_isolates_homophily_agreement_effect(tmp_path, monkeypatch):
    """Homophily's whole story is stronger stance agreement with what a user
    consumes; that should show up as a positive agreement_effect against the
    matched null, averaged over several seeds.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    base = _base_cfg(n_users=400, n_ticks=6)
    cells = build_experiment1(base, kernels=("homophily",), rankers=("affinity",))

    rows = run_experiment1(cells, seeds=list(range(10)))
    summary = summarize_experiment1(rows)
    agreement_effect_mean = summary["agreement_effect_mean"][0]
    assert agreement_effect_mean > 0
