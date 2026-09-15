"""Conformance tests for work-order-01's decision 1: breaking the V7.4
`affect_d0`/`d_cross` tie.

House style (tests/test_change_spec_v7.py): each test observes the
mechanism's *effect*, never its definition. `d_cross`'s own calibration
(camp-boundary, NaN below the gate) lives in tests/test_experiment03.py,
next to the outcome-pair code it extends -- this file covers only the
LIVE engine-side half: `phi`'s own `d0`, calibrated per-population.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.runner import phase_rngs


def _cfg(n_users: int = 400, n_ticks: int = 12, **dyn):
    return dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=n_users, **dyn.pop("pop", {})),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=n_ticks, **dyn),
    )


def _engine(cfg, seed: int = 0):
    from discourse_lab.dynamics.tick import TickEngine
    from discourse_lab.network import cached_graph
    from discourse_lab.population import cached_population

    rngs = phase_rngs(seed)
    pop = cached_population(cfg, seed, rngs["population"])
    graph = cached_graph(cfg, seed, pop, rngs["graph"])
    return TickEngine(cfg=cfg, pop=pop, graph=graph, rngs=rngs)


def test_affect_d0_calibrated_tracks_the_populations_own_median_pairwise_distance():
    """Decision 1a's own acceptance criterion #1: `d0` should track the
    population's median pairwise stance distance, not a constant carried
    over from a different dimensionality. Checked against an INDEPENDENTLY
    computed full pairwise median (not the engine's own 20k-pair sample),
    at a population small enough that the full median is cheap -- the two
    should agree closely even though one is exact and the other sampled.
    """
    cfg = _cfg(n_users=300, stance_polarization=6.0)
    engine = _engine(cfg)

    stance_cols = [i for i, n in enumerate(engine.pop.trait_names) if n.startswith("stance_")]
    stance = engine.pop.X_used[:, stance_cols]
    n = stance.shape[0]
    full_pairwise = np.linalg.norm(stance[:, None, :] - stance[None, :, :], axis=-1)
    true_median = float(np.median(full_pairwise[np.triu_indices(n, k=1)]))

    assert engine.affect_d0_calibrated == pytest.approx(true_median, rel=0.05)
    # deliberately the SAME statistic as agree_delta (B3: two consumers of
    # one population number), not independently re-sampled and coincidentally close
    assert engine.affect_d0_calibrated == engine.agree_delta


def test_affect_d0_mode_switch_has_a_real_effect_on_the_running_engine():
    """The escape hatch decision 1a asks for, observed as an effect rather
    than asserted from the definition: stepping the SAME population under
    "fixed" (pinned at the literal `affect_d0=1.0`) vs "calibrated" (this
    population's own, much larger, median pairwise distance) must produce
    DIFFERENT animus trajectories -- if they matched, the mode switch would
    not be wired to anything, and "fixed" would not be reproducing the
    pre-V8 constant so much as silently always calibrating anyway.
    """
    from discourse_lab.runner import run_iter

    base = _cfg(n_users=300, n_ticks=15, stance_polarization=6.0, affect_drive="distance", pop={"affect": True})
    fixed_cfg = dataclasses.replace(base, dynamics=dataclasses.replace(base.dynamics, affect_d0_mode="fixed", affect_d0=1.0))
    calibrated_cfg = dataclasses.replace(base, dynamics=dataclasses.replace(base.dynamics, affect_d0_mode="calibrated"))

    engine = _engine(calibrated_cfg)
    assert engine.affect_d0_calibrated > 2.0, (
        "test setup: this population's calibrated d0 should be well above the "
        f"fixed 1.0 for the two modes to be distinguishable, got {engine.affect_d0_calibrated}"
    )

    def _final_mean_abs_animus(cfg):
        from discourse_lab.population.traits import trait_names

        animus_col = trait_names(cfg).index("animus")
        final_state = None
        for state in run_iter(cfg, seed=0):
            final_state = state
        return float(np.abs(final_state.traits_snapshot[:, animus_col]).mean())

    fixed_result = _final_mean_abs_animus(fixed_cfg)
    calibrated_result = _final_mean_abs_animus(calibrated_cfg)
    assert fixed_result != pytest.approx(calibrated_result, rel=1e-6), (
        "fixed and calibrated modes produced identical trajectories -- "
        "affect_d0_mode is not actually wired into the running engine"
    )


def test_affect_d0_calibrated_is_not_equal_to_d_cross():
    """Decision 1c's third pin: the two constants must NOT be equal, so a
    future change cannot silently re-tie them (V7.4's original coupling).
    `d0` is the population's own median pairwise distance (a "how far
    apart are two random users, typically" statistic); `d_cross`
    (tests/test_experiment03.py) is the midpoint between the SAME-camp and
    CROSS-camp classes' own mean distances -- a different statistic by
    construction once camps exist, not just a different number by luck.
    """
    from discourse_lab.experiments.experiment03_bubble_intervention import (
        _camp_boundary_d_cross,
        _camp_split_from_stance0,
    )

    cfg = _cfg(n_users=400, stance_polarization=6.0)
    engine = _engine(cfg)
    stance_cols = [i for i, n in enumerate(engine.pop.trait_names) if n.startswith("stance_")]
    stance = engine.pop.X_used[:, stance_cols]
    camp0, bimodality0 = _camp_split_from_stance0(stance)
    assert camp0 is not None, "test setup: this population must be bimodal enough to have camps"

    d_cross = _camp_boundary_d_cross(stance, camp0)
    assert engine.affect_d0_calibrated != pytest.approx(d_cross, rel=0.05), (
        f"d0 ({engine.affect_d0_calibrated}) and d_cross ({d_cross}) must not "
        "coincide -- that would silently reintroduce the V7.4 tie this decision breaks"
    )
