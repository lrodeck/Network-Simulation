"""Step 10 verification (dev §6): sweep kernels/rankers over multiple seeds
(min 10) via the flat seed-major queue; every effect measured against its
matched kernel="null" run.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from discourse_lab.config import Config
from discourse_lab.experiments import (
    DEFAULT_KERNELS,
    DEFAULT_RANKERS,
    build_experiment1,
    run_experiment1,
    summarize_experiment1,
)


def _base_cfg(n_users=300, n_ticks=6):
    # Experiment 1 isolates kernel/ranker effects; drift is step 11's own
    # subject and would otherwise confound the null comparison.
    return dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=n_users),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=n_ticks, drift="none"),
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

    Sized for statistical power, not for speed. The effect is small — a few
    hundredths of a stance unit — and this ran at n_users=400, n_ticks=6,
    10 seeds, where it is indistinguishable from noise:

        n=400 ticks=6  seeds=10:  mean=-0.0009  sd=0.0141  t=-0.21
        n=800 ticks=20 seeds=20:  mean=+0.0258  sd=0.0306  t=+3.77

    At the small size the assertion was passing on the sign of a coin flip,
    which is why it broke on a calibration change that did not touch the
    mechanism it tests. The t-statistic, not the bare sign, is what carries
    the claim, so assert on that.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    base = _base_cfg(n_users=800, n_ticks=20)
    cells = build_experiment1(base, kernels=("homophily",), rankers=("affinity",))

    seeds = list(range(20))
    rows = run_experiment1(cells, seeds=seeds)
    summary = summarize_experiment1(rows)

    effects = np.array([r["agreement_effect"] for r in rows])
    mean = summary["agreement_effect_mean"][0]
    t_stat = mean / (effects.std(ddof=1) / np.sqrt(len(seeds)))

    assert mean > 0, f"agreement effect {mean:+.4f} is not positive"
    assert t_stat > 2.0, f"agreement effect {mean:+.4f} is not distinguishable from noise (t={t_stat:.2f})"
