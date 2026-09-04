"""Step 1 verification (dev §6): caching, artifact invalidation, phase RNG discipline."""

from __future__ import annotations

import dataclasses
import os

import numpy as np

from discourse_lab.config import Config, PopulationConfig
from discourse_lab.io.artifacts import graph_key, population_key
from discourse_lab.runner import PHASES, cached_run, load_run, phase_rngs, run


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
