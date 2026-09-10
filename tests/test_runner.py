"""Step 1 verification (dev §6): caching, artifact invalidation, phase RNG discipline."""

from __future__ import annotations

import dataclasses
import os

import numpy as np

from discourse_lab.config import Config, DynamicsConfig, PopulationConfig, effective_dynamics
from discourse_lab.io.artifacts import graph_key, population_key
from discourse_lab.runner import PHASES, cached_run, load_run, phase_rngs, run, run_iter


def test_run_caches_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = dataclasses.replace(Config(), dynamics=dataclasses.replace(Config().dynamics, n_ticks=3))

    path1 = cached_run(cfg, seed=0)
    assert (path1 / "COMPLETE").exists()
    mtime1 = (path1 / "COMPLETE").stat().st_mtime_ns

    path2 = cached_run(cfg, seed=0)
    assert path2 == path1
    assert (path2 / "COMPLETE").stat().st_mtime_ns == mtime1  # not rerun

    handle = load_run(cfg, seed=0)
    assert handle.complete
    assert len(handle.ticks()) == 3


def test_population_field_invalidates_only_population_artifact():
    base = Config()
    changed_pop = dataclasses.replace(
        base, population=dataclasses.replace(base.population, n_users=base.population.n_users + 1)
    )

    assert population_key(base) != population_key(changed_pop)
    assert graph_key(base) != graph_key(changed_pop)  # depends on population

    # a dynamics-only change must not touch population or graph keys
    changed_dyn = dataclasses.replace(
        base, dynamics=dataclasses.replace(base.dynamics, n_ticks=base.dynamics.n_ticks + 1)
    )
    assert population_key(base) == population_key(changed_dyn)
    assert graph_key(base) == graph_key(changed_dyn)


def test_phase_rng_streams_stable_and_independent():
    rngs_a = phase_rngs(seed=42)
    rngs_b = phase_rngs(seed=42)

    for name in PHASES:
        draw_a = rngs_a[name].standard_normal(8)
        draw_b = rngs_b[name].standard_normal(8)
        np.testing.assert_array_equal(draw_a, draw_b)  # stable across calls, same seed

    fresh = phase_rngs(seed=42)
    draws = {name: fresh[name].standard_normal(8) for name in PHASES}
    for i, name_i in enumerate(PHASES):
        for name_j in PHASES[i + 1 :]:
            assert not np.array_equal(draws[name_i], draws[name_j])  # independent streams


def test_effective_dynamics_resolves_the_latest_started_entry():
    base = DynamicsConfig(inject_k=0)
    sched = dataclasses.replace(
        base, schedule=((5, (("inject_k", 20),)), (10, (("inject_k", 0),)))
    )
    assert effective_dynamics(sched, 0) is sched          # no entry started: identity, no copy
    assert effective_dynamics(sched, 4).inject_k == 0
    assert effective_dynamics(sched, 5).inject_k == 20
    assert effective_dynamics(sched, 9).inject_k == 20
    assert effective_dynamics(sched, 10).inject_k == 0     # withdrawal restates, not undoes
    assert effective_dynamics(base, 999) is base           # empty schedule: always identity


def test_schedule_gives_a_bit_identical_prefix_and_diverges_after(monkeypatch, tmp_path):
    """Experiment 03 §5.1's package prerequisite, proved rather than assumed:
    two configs sharing a seed and an identical dynamics config up to tick T,
    differing only in a `schedule` entry AT T, must produce per-tick metrics
    that are bit-identical for t < T (same RNG draws, same effective params)
    and diverge from T onward (the arm is actually doing something). This is
    the property the whole fork design in Experiment 03's design doc depends
    on — separate configs that "merely start the same way" would not give it.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    T = 8
    base = dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=200),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=T + 6, inject_k=0),
    )
    none_arm = base
    intervention_arm = dataclasses.replace(
        base,
        dynamics=dataclasses.replace(
            base.dynamics, schedule=((T, (("inject_k", 20),)),)
        ),
    )

    def _value_equal(x, y) -> bool:
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            return True  # nan != nan under `==`, but both mean "ungraded this tick"
        return x == y

    def _tick_equal(a: dict, b: dict) -> bool:
        return a.keys() == b.keys() and all(_value_equal(a[k], b[k]) for k in a)

    metrics_none = [s.metrics for s in run_iter(none_arm, seed=0)]
    metrics_intv = [s.metrics for s in run_iter(intervention_arm, seed=0)]

    for t in range(T):
        assert _tick_equal(metrics_none[t], metrics_intv[t]), f"prefix diverged at t={t}, before T={T}"

    diverged = any(
        not _tick_equal(metrics_none[t], metrics_intv[t]) for t in range(T, len(metrics_none))
    )
    assert diverged, "arms never diverged after the schedule boundary — inject_k=20 did nothing"
