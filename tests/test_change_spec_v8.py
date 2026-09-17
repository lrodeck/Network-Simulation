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
    cfg = _cfg(n_users=300, pop={"stance_polarization": 6.0})
    engine = _engine(cfg)
    assert cfg.dynamics.agreement_metric == "rms", "test assumes the default metric"

    stance_cols = [i for i, n in enumerate(engine.pop.trait_names) if n.startswith("stance_")]
    stance = engine.pop.X_used[:, stance_cols]
    n = stance.shape[0]
    full_pairwise = np.linalg.norm(stance[:, None, :] - stance[None, :, :], axis=-1)
    # RMS-normalized (/sqrt(D)), matching agree_delta's own convention -- the
    # SAME units `stance_distance` (-features["agreement"]) is actually in
    # wherever agreement_metric="rms", which is what phi(d) consumes.
    true_median = float(np.median(full_pairwise[np.triu_indices(n, k=1)])) / np.sqrt(len(stance_cols))

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

    base = _cfg(n_users=300, n_ticks=15, affect_drive="distance", pop={"affect": True, "stance_polarization": 6.0})
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
    apart are two random users, typically" statistic, mixing same- and
    cross-camp pairs); `d_cross` (tests/test_experiment03.py) is the
    midpoint between the SAME-camp and CROSS-camp classes' own mean
    distances -- a different statistic by construction.

    On the ACTUAL Experiment 03 population generator (checked directly --
    a `stance_polarization=X` fixture at this file's own reduced scale
    lands the two within ~1-5% of each other, since only axis 0 carries
    camp signal and 2 undifferentiated noise axes dominate total distance
    for BOTH statistics at small N; `dial_config`'s full 3-dial setup at
    N=1000 separates them somewhat more, ~7-10%), so this test uses THAT
    generator rather than the file's own lighter `_cfg` fixture.
    """
    from discourse_lab.experiments.experiment03_bubble_intervention import (
        _camp_boundary_d_cross,
        _camp_split_from_stance0,
        base_config,
        dial_config,
        run_arm,
        _stance_and_animus_at,
    )

    burn_in = dial_config(base_config(1000, 60), affective=0.7, ideological=0.7, structural=0.7)
    none_cfg, handle_none = run_arm(burn_in, "none", 0, intervention_tick=59, n_ticks_total=60)
    stance0, _ = _stance_and_animus_at(handle_none, none_cfg, 58)
    camp0, _ = _camp_split_from_stance0(stance0)
    assert camp0 is not None, "test setup: this population must be bimodal enough to have camps"

    engine = _engine(none_cfg)
    d_cross = _camp_boundary_d_cross(stance0, camp0, rms=True)
    assert engine.affect_d0_calibrated != pytest.approx(d_cross, rel=0.03), (
        f"d0 ({engine.affect_d0_calibrated}) and d_cross ({d_cross}) must not "
        "coincide -- that would silently reintroduce the V7.4 tie this decision breaks"
    )


# --------------------------------------------------------------------------
# V8.4: phi's sigmoid shape (change-spec-v8.md's decision (a)) -- a sigmoid
# centred on the population's own median pairwise distance, width from its
# IQR, replacing the saturating `d / (d + d0)` shape that is floor-bounded
# below by `d_same/d_cross` for ANY `d0` (the spec's own stated constraint).
# Adopted after the audit below confirmed the premise on an actual
# Experiment-03-scale population, same audit shape as the d_cross pin above.
# --------------------------------------------------------------------------


def test_phi_sigmoid_is_centred_at_affect_d0():
    """`phi(d0) == 0.5` by construction (`expit(0) == 0.5`) -- the sigmoid's
    centre is a LOCATION, unlike the saturating shape's `d0` which is a
    saturation rate; this is the property the whole decision leans on."""
    from discourse_lab.dynamics.drift import affect_delta
    from discourse_lab.dynamics.valence import assign_valence
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=200, pop={"affect": True})
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    from discourse_lab.dynamics.expression import ExpressionMap
    from discourse_lab.dynamics.posts import generate_posts

    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    posts = generate_posts(np.arange(4) % 200, pop, expr, np.zeros(K), np.zeros((K, D)), 0.3, rng)

    m = 100
    exposures = Exposures(
        post_idx=np.zeros(m, dtype=int), user_id=(np.arange(m) % 99) * 2 + 1,
        rank=np.zeros(m, dtype=int), is_follower=np.ones(m, dtype=bool),
    )
    actions = np.full(m, "like")  # no valence sign complications: hostility weight is action-only here
    valence = assign_valence(np.zeros(m), 0.0, rng, force_agree=True, force_civil=True)
    animus_col = pop.trait_names.index("animus")

    d0 = 2.5
    stance_distance = np.full(m, d0)  # every exposure sits EXACTLY at the centre
    delta = affect_delta(
        exposures, actions, posts, pop, camps=None, lr_affect=1.0, valence=valence,
        mode="distance", stance_distance=stance_distance, affect_d0=d0,
        phi_shape="sigmoid", phi_width=1.0,
    )
    # outgroup == 0.5 at the centre, so animus moves at exactly HALF the
    # lr_affect*hostility_weight it would at outgroup==1 -- check against the
    # ingroup (identification) term, which uses (1-outgroup)==0.5 too, so the
    # two per-user contributions must be equal here (same weight tables'
    # magnitude structure aside, the point is 0.5/0.5, not 0/1 or 1/0)
    ident_col = pop.trait_names.index("identification")
    touched = np.unique(exposures.user_id)
    assert (delta[touched, animus_col] != 0.0).all()
    assert (delta[touched, ident_col] != 0.0).all()


def test_phi_sigmoid_saturates_toward_0_and_1_away_from_centre():
    from discourse_lab.dynamics.drift import affect_delta
    from discourse_lab.dynamics.valence import assign_valence
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.dynamics.expression import ExpressionMap
    from discourse_lab.dynamics.posts import generate_posts
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=200, pop={"affect": True})
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    posts = generate_posts(np.arange(4) % 200, pop, expr, np.zeros(K), np.zeros((K, D)), 0.3, rng)

    m = 100
    exposures = Exposures(
        post_idx=np.zeros(m, dtype=int), user_id=(np.arange(m) % 99) * 2 + 1,
        rank=np.zeros(m, dtype=int), is_follower=np.ones(m, dtype=bool),
    )
    actions = np.full(m, "like")
    valence = assign_valence(np.zeros(m), 0.0, rng, force_agree=True, force_civil=True)
    animus_col = pop.trait_names.index("animus")
    touched = np.unique(exposures.user_id)
    d0, width = 2.5, 0.1  # narrow width: far-from-centre distances saturate hard

    near_delta = affect_delta(
        exposures, actions, posts, pop, camps=None, lr_affect=1.0, valence=valence,
        mode="distance", stance_distance=np.full(m, d0 - 5 * width), affect_d0=d0,
        phi_shape="sigmoid", phi_width=width,
    )
    far_delta = affect_delta(
        exposures, actions, posts, pop, camps=None, lr_affect=1.0, valence=valence,
        mode="distance", stance_distance=np.full(m, d0 + 5 * width), affect_d0=d0,
        phi_shape="sigmoid", phi_width=width,
    )
    # far below the centre: outgroup ~ 0, so animus (driven by outgroup) is
    # ~0 while identification (driven by ingroup = 1-outgroup ~ 1) is not
    assert np.abs(near_delta[touched, animus_col]).max() < 1e-2
    assert np.abs(near_delta[touched, ident_col := pop.trait_names.index("identification")]).min() > 1e-2
    # far above the centre: the reverse
    assert np.abs(far_delta[touched, animus_col]).min() > 1e-2
    assert np.abs(far_delta[touched, ident_col]).max() < 1e-2


def test_phi_saturating_shape_is_the_unchanged_pre_v8_4_default():
    """`affect_phi_shape="saturating"` (the config default) must reproduce
    `d / (d + d0)` exactly -- V8.4 must not silently change behaviour for
    every run that has not opted into `"sigmoid"`."""
    from discourse_lab.dynamics.drift import affect_delta
    from discourse_lab.dynamics.valence import assign_valence
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.dynamics.expression import ExpressionMap
    from discourse_lab.dynamics.posts import generate_posts
    from discourse_lab.population import sample_population

    assert Config().dynamics.affect_phi_shape == "saturating", "test setup: this must stay the shipped default"

    cfg = _cfg(n_users=200, pop={"affect": True})
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    posts = generate_posts(np.arange(4) % 200, pop, expr, np.zeros(K), np.zeros((K, D)), 0.3, rng)

    m = 100
    exposures = Exposures(
        post_idx=np.zeros(m, dtype=int), user_id=(np.arange(m) % 99) * 2 + 1,
        rank=np.zeros(m, dtype=int), is_follower=np.ones(m, dtype=bool),
    )
    actions = np.full(m, "reply")
    valence = assign_valence(np.zeros(m), 0.0, rng, force_agree=False, force_civil=False)
    d0 = 1.7
    stance_distance = rng.uniform(0.0, 5.0, size=m)

    delta_default = affect_delta(
        exposures, actions, posts, pop, camps=None, lr_affect=0.5, valence=valence,
        mode="distance", stance_distance=stance_distance, affect_d0=d0,
    )
    delta_explicit = affect_delta(
        exposures, actions, posts, pop, camps=None, lr_affect=0.5, valence=valence,
        mode="distance", stance_distance=stance_distance, affect_d0=d0,
        phi_shape="saturating",
    )
    np.testing.assert_array_equal(delta_default, delta_explicit)


def test_unknown_phi_shape_raises():
    from discourse_lab.dynamics.drift import affect_delta
    from discourse_lab.dynamics.valence import assign_valence
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=50, pop={"affect": True})
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    m = 10
    exposures = Exposures(
        post_idx=np.zeros(m, dtype=int), user_id=np.arange(m),
        rank=np.zeros(m, dtype=int), is_follower=np.ones(m, dtype=bool),
    )
    actions = np.full(m, "like")
    valence = assign_valence(np.zeros(m), 0.0, rng, force_agree=True, force_civil=True)
    with pytest.raises(ValueError, match="affect_phi_shape"):
        affect_delta(
            exposures, actions, None, pop, camps=None, lr_affect=0.5, valence=valence,
            mode="distance", stance_distance=np.ones(m), affect_d0=1.0, phi_shape="bogus",
        )


def test_affect_phi_width_calibrated_tracks_the_populations_own_iqr():
    """V8.4(a)'s width, like decision 1a's `d0`, must track this
    population's own geometry, not a constant carried over from elsewhere --
    checked against an INDEPENDENTLY computed full-population IQR."""
    cfg = _cfg(n_users=300, pop={"stance_polarization": 6.0})
    engine = _engine(cfg)

    stance_cols = [i for i, n in enumerate(engine.pop.trait_names) if n.startswith("stance_")]
    stance = engine.pop.X_used[:, stance_cols]
    n = stance.shape[0]
    full_pairwise = np.linalg.norm(stance[:, None, :] - stance[None, :, :], axis=-1)
    d = full_pairwise[np.triu_indices(n, k=1)] / np.sqrt(len(stance_cols))
    true_iqr = float(np.percentile(d, 75) - np.percentile(d, 25))

    assert engine.affect_phi_width_calibrated == pytest.approx(true_iqr, rel=0.1)


def test_phi_sigmoid_separates_realized_camps_on_an_experiment03_scale_population():
    """The audit that resolved V8.4's own open fork (change-spec-v8.md):
    checked directly on the Experiment 03 population generator at its own
    design scale (n_users=1000, ideological=0.9 -- one of wave_a_prime_
    recal's own swept design points) rather than assumed from the algebra.
    A threshold at the population's own median pairwise distance must
    separate realized same-camp from cross-camp pairs well above chance —
    this is the empirical premise decision (a) rests on, and it is what
    actually failed for `"saturating"` (bounded below by `d_same/d_cross`
    regardless of calibration, per the shape argument in `affect_delta`'s
    own docstring)."""
    from discourse_lab.experiments.experiment03_bubble_intervention import (
        _camp_split_from_stance0,
        _stance_and_animus_at,
        base_config,
        dial_config,
        run_arm,
    )

    n_ticks_burn_in = 60
    burn_in = dial_config(base_config(1000, n_ticks_burn_in), affective=0.5, ideological=0.9, structural=0.5)
    none_cfg, handle_none = run_arm(burn_in, "none", 0, intervention_tick=n_ticks_burn_in, n_ticks_total=n_ticks_burn_in)
    stance0, _ = _stance_and_animus_at(handle_none, none_cfg, n_ticks_burn_in - 1)
    camp0, bimodality0 = _camp_split_from_stance0(stance0)
    assert camp0 is not None, "test setup: this design point must be bimodal enough to have camps"

    rng = np.random.default_rng(0)
    n = stance0.shape[0]
    n_sample = min(20_000, n * 4)
    a = rng.integers(0, n, n_sample)
    b = rng.integers(0, n, n_sample)
    dist = np.linalg.norm(stance0[a] - stance0[b], axis=1) / np.sqrt(stance0.shape[1])
    median_d = float(np.median(dist))

    same = camp0[a] == camp0[b]
    cross = ~same
    same_below = (dist[same] < median_d).mean()
    cross_above = (dist[cross] > median_d).mean()
    separation_accuracy = (same_below + cross_above) / 2.0

    assert separation_accuracy > 0.85, (
        f"median-centred separation was only {separation_accuracy:.3f} at a clearly "
        f"bimodal design point (bimodality={bimodality0:.3f}) -- the empirical premise "
        "V8.4 decision (a) rests on would not hold here"
    )
