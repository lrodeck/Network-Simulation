"""Conformance tests for change-spec-v7-continuous-affect.md (V7.1-V7.6).

House style (tests/test_change_spec.py): each test observes the mechanism's
*effect*, never its definition. Experiment-03-level items (V7.2, V7.4, V7.5's
`delta_bic_margin` plumbing) live in tests/test_experiment03.py instead,
next to the outcome-pair code they extend.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.runner import phase_rngs, run_iter


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


def _tiny_posts(cfg, rng):
    from discourse_lab.dynamics.expression import ExpressionMap
    from discourse_lab.dynamics.posts import generate_posts
    from discourse_lab.population import sample_population

    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    return generate_posts(np.arange(4) % 200, pop, expr, np.zeros(K), np.zeros((K, D)), 0.3, rng)


# --------------------------------------------------------------------------
# V7.1: record gate state
# --------------------------------------------------------------------------


def test_v7_1_affect_gated_true_throughout_below_the_gate():
    """A population below the Sarle bimodality gate (default
    stance_polarization=0, unimodal) must report `affect_gated=True` on
    every tick, and every tick's `bimodality` reading must sit at or below
    the 5/9 threshold (or be unmeasurable) -- the exact coupling V7.1 exists
    to make auditable rather than inferred from a suspiciously-uniform
    delta_aff_plateau column."""
    cfg = _cfg(
        n_users=400, n_ticks=15, pop={"affect": True},
        kernel="outrage", drift="full", drift_ramp_ticks=2, affect_drive="camp",
    )
    rows = [s.metrics for s in run_iter(cfg, seed=0)]
    assert rows, "no ticks produced"
    assert all(r["affect_gated"] is True for r in rows), (
        f"a below-gate run did not report affect_gated=True on every tick: "
        f"{[r['affect_gated'] for r in rows]}"
    )
    assert all(not np.isfinite(r["bimodality"]) or r["bimodality"] <= 5 / 9 for r in rows), (
        f"bimodality exceeded the gate on a nominally-unimodal population: "
        f"{[r['bimodality'] for r in rows]}"
    )


def test_v7_1_affect_gated_false_throughout_above_the_gate():
    """A strongly bimodal population must report `affect_gated=False` on
    every tick once camps are defined (bimodality does not cross the gate
    mid-run at this separation)."""
    cfg = _cfg(
        n_users=400, n_ticks=15, pop={"affect": True, "stance_polarization": 6.0},
        kernel="outrage", drift="full", drift_ramp_ticks=2, affect_drive="camp",
    )
    rows = [s.metrics for s in run_iter(cfg, seed=0)]
    assert rows, "no ticks produced"
    gated_ticks = [i for i, r in enumerate(rows) if r["affect_gated"] is True]
    assert not gated_ticks, (
        f"an above-gate run reported affect_gated=True at ticks {gated_ticks}: "
        f"{[r['bimodality'] for r in rows]}"
    )
    assert all(np.isfinite(r["bimodality"]) and r["bimodality"] > 5 / 9 for r in rows[1:]), (
        f"bimodality did not read above the gate once camps were defined: "
        f"{[r['bimodality'] for r in rows]}"
    )


# --------------------------------------------------------------------------
# V7.3: animus drive keyed on continuous stance distance
# --------------------------------------------------------------------------


def test_v7_3_distance_mode_moves_animus_where_camp_mode_is_gated_off():
    """V7.3's core claim, isolated from simulation noise the way the V5/V1-V2
    tests isolate `affect_delta`: on an exposure batch where `camps` is None
    (exactly the below-the-gate case), `affect_drive="camp"` must produce
    zero animus movement -- the coupling this whole change spec exists to
    fix -- while `"distance"` must move it from the SAME exposures, using
    continuous per-axis stance distance instead of a camp label."""
    from discourse_lab.dynamics.drift import affect_delta
    from discourse_lab.dynamics.valence import assign_valence
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=200, pop={"affect": True})
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    posts = _tiny_posts(cfg, rng)

    m = 100
    exposures = Exposures(
        post_idx=np.zeros(m, dtype=int),
        user_id=(np.arange(m) % 99) * 2 + 1,
        rank=np.zeros(m, dtype=int),
        is_follower=np.ones(m, dtype=bool),
    )
    actions = np.full(m, "reply")
    valence = assign_valence(np.zeros(m), 0.0, rng, force_agree=False, force_civil=False)  # disagree+hostile
    animus_col = pop.trait_names.index("animus")
    touched = np.unique(exposures.user_id)

    delta_camp = affect_delta(
        exposures, actions, posts, pop, camps=None, lr_affect=0.5, valence=valence, mode="camp",
    )
    assert (delta_camp[:, animus_col] == 0.0).all(), (
        "camp mode moved animus despite camps=None -- the bimodality gate is not wired"
    )

    stance_cols = [i for i, n in enumerate(pop.trait_names) if n.startswith("stance_")]
    user_stance = pop.X_used[exposures.user_id][:, stance_cols]
    post_stance = posts.stance[exposures.post_idx]
    stance_distance = np.linalg.norm(user_stance - post_stance, axis=1)

    delta_distance = affect_delta(
        exposures, actions, posts, pop, camps=None, lr_affect=0.5, valence=valence,
        mode="distance", stance_distance=stance_distance, affect_d0=1.0,
    )
    assert (delta_distance[touched, animus_col] != 0.0).all(), (
        "distance mode produced no animus movement even though camps=None -- the point of the change"
    )


def test_v7_3_camp_and_distance_modes_correlate_above_0_95_when_bimodal():
    """On a population ALREADY bimodal enough for `"camp"` mode to work, the
    two mechanisms must not silently diverge: fed the IDENTICAL exposure
    batch (same actions, same valence), their per-user animus trajectories
    must correlate above 0.95, so V7.3 does not rewrite behaviour in the
    region Experiment 01's SBM finding already measured.

    Run through the full tick engine rather than `affect_delta` directly:
    under `kernel="outrage"` animus feeds back into the reaction phase via
    `outgroup_x_animus` (the `report` feature), so ANY difference between
    the two modes' animus values -- however small -- can tip a downstream
    action draw and chaotically decorrelate the rest of the same-seed run.
    Isolating the mechanism (matching this file's other V7.3 test) is the
    right check for "does the formula agree"; this one is the right check
    for "does wiring it into the live feedback loop still agree", so both
    are kept.
    """
    trajectories = {}
    for mode in ("camp", "distance"):
        cfg = _cfg(
            n_users=500, n_ticks=60, pop={"affect": True, "stance_polarization": 6.0},
            kernel="outrage", drift="full", drift_ramp_ticks=5, affect_drive=mode,
        )
        engine = _engine(cfg)
        engine._refresh_camps()
        assert engine.camps is not None, "test setup: population must be bimodal"
        for t in range(cfg.dynamics.n_ticks):
            engine.step(t)
        trajectories[mode] = engine.pop.animus.copy()

    corr = float(np.corrcoef(trajectories["camp"], trajectories["distance"])[0, 1])
    assert corr > 0.95, (
        f"camp vs distance final per-user animus correlate at {corr:.3f} on a bimodal "
        f"population, expected > 0.95"
    )


def test_v7_3_affect_delta_formulas_correlate_above_0_95_when_bimodal_given_identical_inputs():
    """The mechanism-level companion to the full-engine test above: fed the
    SAME exposures/actions/valence (no feedback loop, no RNG divergence
    possible) and a dyad-distance array that CLEANLY tracks camp membership
    -- small within a camp, large across -- camp mode's binary outgroup
    indicator and distance mode's continuous phi(d) must produce
    near-identical per-user animus deltas.

    A hand-built `camps` array and a hand-built `stance_distance` array
    (rather than deriving both from one `sample_population` draw, as the
    full-engine test above does) isolate the two FORMULAS from a
    population's own within-mode spread: `bimodal_normal`'s two modes carry
    enough of their own variance, on top of `affect_d0`'s fixed d0=1.0
    calibration for a one-axis population, that a handful of same-camp
    outliers can sit at a phi(d) not so different from a typical
    cross-camp pair -- a real scale property of a fixed constant against a
    realistic marginal, not a disagreement between the two mechanisms this
    test exists to check.
    """
    from discourse_lab.dynamics.drift import affect_delta
    from discourse_lab.dynamics.valence import assign_valence
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=200, pop={"affect": True})
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    posts = _tiny_posts(cfg, rng)  # 4 posts, authors 0-3

    camps = (np.arange(cfg.population.n_users) % 2).astype(np.int64)
    assert list(camps[:4]) == [0, 1, 0, 1]  # the 4 post authors span both camps

    m = 4000
    rng2 = np.random.default_rng(1)
    post_idx = rng2.integers(0, 4, m)
    user_id = rng2.integers(0, cfg.population.n_users, m)
    exposures = Exposures(
        post_idx=post_idx, user_id=user_id, rank=np.zeros(m, dtype=int), is_follower=np.ones(m, dtype=bool),
    )
    actions = rng2.choice(["like", "reply", "repost", "quote"], size=m)
    valence = assign_valence(np.zeros(m), 0.0, rng2, force_agree=False, force_civil=False)

    outgroup = (camps[user_id] != camps[posts.author[post_idx]])
    stance_distance = np.where(outgroup, rng2.uniform(5.0, 6.0, m), rng2.uniform(0.0, 0.3, m))

    delta_camp = affect_delta(
        exposures, actions, posts, pop, camps=camps, lr_affect=0.5, valence=valence, mode="camp",
    )
    delta_distance = affect_delta(
        exposures, actions, posts, pop, camps=camps, lr_affect=0.5, valence=valence,
        mode="distance", stance_distance=stance_distance, affect_d0=1.0,
    )

    animus_col = pop.trait_names.index("animus")
    touched = np.unique(user_id)
    corr = float(np.corrcoef(delta_camp[touched, animus_col], delta_distance[touched, animus_col])[0, 1])
    assert corr > 0.95, (
        f"camp vs distance per-user animus deltas correlate at {corr:.3f} when distance "
        f"cleanly tracks camp membership, expected > 0.95"
    )


def test_v7_3_unknown_affect_drive_mode_raises():
    from discourse_lab.dynamics.drift import affect_delta
    from discourse_lab.dynamics.valence import assign_valence
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=50, pop={"affect": True})
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    posts = _tiny_posts(cfg, rng)
    exposures = Exposures(
        post_idx=np.zeros(5, dtype=int), user_id=np.arange(5),
        rank=np.zeros(5, dtype=int), is_follower=np.ones(5, dtype=bool),
    )
    actions = np.full(5, "like")
    valence = assign_valence(np.zeros(5), 0.0, rng, force_agree=True, force_civil=True)

    with pytest.raises(ValueError):
        affect_delta(exposures, actions, posts, pop, camps=None, lr_affect=0.5, valence=valence, mode="bogus")


# --------------------------------------------------------------------------
# V7.6: re-gate the substrate
# --------------------------------------------------------------------------


def test_v7_6_gate_rows_is_reciprocity_only():
    from discourse_lab.experiments.gate import GATE_ROWS

    assert GATE_ROWS == ("reciprocity",), (
        "attention_gini must be a reported diagnostic, not a blocking gate row -- "
        "it is kernel-bound (in-band only under bandwagon) and unreachable under "
        "Experiment 03's own outrage kernel"
    )


def test_v7_6_reciprocity_gate_passes_at_the_experiment03_substrate_both_scales():
    """V7.6's own test: the Wave-A-background Experiment 03 substrate, under
    the now-default `affect_drive="distance"`, must pass the reciprocity
    gate at both N=2,000 and N=10,000. `attention_gini`/`clustering_ratio`
    are still measured (and would be flagged against their reference range
    by `stylized_facts_report`), just not blocking.

    This substrate had never actually been run through `stylized_gate`
    before V7.6 -- doing so here is what surfaced `graph.sbm_mirror_p`'s
    0.0 default leaving SBM reciprocity at ~0.065 (FINDINGS.md), well under
    band; `dial_config` now sets 0.15, calibrated empirically the same way
    `experiments/gate.py::calibrated_gate_config` calibrated its own
    `mirror_p`. Both scales are exercised in one test (rather than two) so a
    failure at one scale doesn't look like an unrelated flake at the other.

    Short `n_ticks` and few seeds: N=10,000 costs real wall-clock time per
    seed (~100s at this substrate) even at this length. `DLAB_GATE_SEEDS_N10K`
    widens the N=10,000 leg for a fuller check without slowing routine runs.
    """
    import os

    from discourse_lab.experiments.experiment03_bubble_intervention import base_config, dial_config
    from discourse_lab.experiments.gate import stylized_gate

    small = dial_config(base_config(n_users=2000, n_ticks=200), affective=0.7, ideological=0.7, structural=0.7)
    assert small.dynamics.affect_drive == "distance"
    report_small = stylized_gate(small, seeds=(0, 1), n_ticks=15)
    assert report_small.passed, f"N=2000: {report_small.summary()}"

    n_seeds_large = int(os.environ.get("DLAB_GATE_SEEDS_N10K", "1"))
    large = dial_config(base_config(n_users=10_000, n_ticks=200), affective=0.7, ideological=0.7, structural=0.7)
    report_large = stylized_gate(large, seeds=tuple(range(n_seeds_large)), n_ticks=15)
    assert report_large.passed, f"N=10000: {report_large.summary()}"

    for report in (report_small, report_large):
        assert "attention_gini" in report.rows, "attention_gini should still be measured, just not blocking"
        assert "clustering_ratio" in report.rows


def test_v7_6_affect_drive_does_not_move_graph_structure_measures():
    """V7.6: "V7 does not touch graph construction, so movement there is a
    leak into the wrong layer" -- reciprocity and clustering_ratio must
    read identically under `"camp"` and `"distance"`, holding everything
    else (including the seed and the cached graph artifact) fixed."""
    from discourse_lab.analysis import set_param
    from discourse_lab.experiments.experiment03_bubble_intervention import base_config, dial_config
    from discourse_lab.experiments.gate import stylized_gate

    base = dial_config(base_config(n_users=2000, n_ticks=200), affective=0.7, ideological=0.7, structural=0.7)
    camp_report = stylized_gate(set_param(base, "dynamics.affect_drive", "camp"), seeds=(0,), n_ticks=15)
    distance_report = stylized_gate(set_param(base, "dynamics.affect_drive", "distance"), seeds=(0,), n_ticks=15)

    assert camp_report.rows["reciprocity"] == pytest.approx(distance_report.rows["reciprocity"])
    assert camp_report.rows["clustering_ratio"] == pytest.approx(distance_report.rows["clustering_ratio"])
