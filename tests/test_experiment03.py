"""Experiment 03 infrastructure: the outcome-pair decomposition math (SS2),
tested on toy stance arrays rather than full runs -- fast, and isolates the
math from simulation noise. The scenarios are chosen to fail the way
Experiment 01's `affective_distance` failed (a raw signed mean of
un-oriented movement cancelling real, opposite-direction movement into a
false zero) if `_ideo_decomposition` regressed to that shape.
"""

from __future__ import annotations

import numpy as np

from discourse_lab.experiments.experiment03_bubble_intervention import (
    DeltaAff,
    DeltaIdeo,
    _aff_from_arrays,
    _fixed_axis,
    _ideo_decomposition,
)


def _two_camps(neg_center: float, pos_center: float, n_per_camp: int = 20, noise: float = 0.15):
    """A clearly bimodal 2-D stance array: two tight clusters on axis 0,
    axis 1 pure noise. `neg_center`/`pos_center` are axis-0 cluster means.
    """
    rng = np.random.default_rng(0)
    neg = np.column_stack([
        rng.normal(neg_center, noise, n_per_camp), rng.normal(0.0, noise, n_per_camp),
    ])
    pos = np.column_stack([
        rng.normal(pos_center, noise, n_per_camp), rng.normal(0.0, noise, n_per_camp),
    ])
    return np.vstack([neg, pos])


def test_toward_mean_does_not_cancel_under_symmetric_convergence():
    """The Experiment 01 lesson, directly: both camps move toward the global
    center by the same amount. A raw signed mean of (stance1 - stance0)
    cancels to exactly zero (the neg camp's +1 and the pos camp's -1 average
    out) -- and would silently read as "no convergence happened". The
    distance-to-mean formulation must not make that mistake.
    """
    stance0 = _two_camps(-3.0, 3.0)
    stance1 = _two_camps(-2.0, 2.0)  # both camps one unit closer to x=0

    naive_signed_mean = (stance1 - stance0)[:, 0].mean()
    assert abs(naive_signed_mean) < 1e-9, "test setup: the naive mean should cancel"

    result = _ideo_decomposition(stance0, stance1, stance0, k1_arm=2, k1_none=2)
    assert result.toward_mean > 0.5, f"convergence should read clearly positive, got {result.toward_mean}"
    # symmetric convergence toward the midpoint between two camps is also,
    # mechanically, movement toward the OTHER camp in this 1-axis setup
    assert result.toward_other_camp > 0.5
    assert result.toward_own_pole < -0.5, "moving toward center is moving AWAY from one's own pole"


def test_toward_own_pole_is_positive_under_symmetric_radicalization():
    """The complementary scenario: both camps move AWAY from center (deeper
    into their own side). All three signed measures should flip relative to
    the convergence test above -- this is what makes them three genuinely
    different lenses on the same displacement, not one measure in disguise.
    """
    stance0 = _two_camps(-3.0, 3.0)
    stance1 = _two_camps(-4.0, 4.0)  # both camps one unit further from x=0

    result = _ideo_decomposition(stance0, stance1, stance0, k1_arm=2, k1_none=2)
    assert result.toward_own_pole > 0.5, f"radicalization should read positive, got {result.toward_own_pole}"
    assert result.toward_mean < -0.5, "moving away from center is moving away from the mean"
    assert result.toward_other_camp < -0.5, "moving away from center is moving away from the other camp"


def test_ideo_components_are_net_of_the_none_arms_own_movement():
    """SS5.3's matched-null discipline applied to the ideological outcome:
    an arm that moves EXACTLY like `none` (e.g. shared OU reversion with no
    intervention effect) must net to zero on all three components, even
    though each arm's own raw movement is large.
    """
    stance0 = _two_camps(-3.0, 3.0)
    stance1_arm = _two_camps(-2.0, 2.0)
    stance1_none = _two_camps(-2.0, 2.0)  # none arm converges identically

    result = _ideo_decomposition(stance0, stance1_arm, stance1_none, k1_arm=2, k1_none=2)
    assert abs(result.toward_mean) < 1e-9
    assert abs(result.toward_other_camp) < 1e-9
    assert abs(result.toward_own_pole) < 1e-9
    assert result.delta_k == 0.0


def test_ideo_components_are_nan_when_pre_period_camp_is_undefined():
    """A unimodal pre-intervention population: camp is a projection artifact,
    not two camps (metrics/polarization.py's own convention), so the three
    camp-relative components must be NaN -- not zero, which would silently
    read as "no ideological movement" rather than "the question is not
    well-posed here". `delta_k` is never gated: V6(2) exists so fragmentation
    is visible precisely where the binary camp frame is not.
    """
    rng = np.random.default_rng(1)
    stance0 = rng.normal(0.0, 1.0, size=(40, 2))  # single blob, no camps
    stance1 = rng.normal(0.0, 1.0, size=(40, 2))

    result = _ideo_decomposition(stance0, stance1, stance0, k1_arm=3, k1_none=1)
    assert np.isnan(result.toward_other_camp)
    assert np.isnan(result.toward_mean)
    assert np.isnan(result.toward_own_pole)
    assert result.delta_k == 2.0  # unaffected by the gate


def test_fixed_axis_projects_a_later_array_onto_the_earlier_frame():
    """`_ideo_decomposition` relies on `_fixed_axis` returning stance0's OWN
    frame so stance1 is scored against a fixed reference rather than its own
    (possibly rotated) principal component."""
    stance0 = _two_camps(-3.0, 3.0)
    mean0, axis0 = _fixed_axis(stance0)
    # the dominant axis of a two-cluster-on-x-axis population points along x
    assert abs(abs(axis0[0]) - 1.0) < 0.05
    assert abs(mean0[0]) < 0.2


def test_aff_plateau_is_arm_minus_none_mean_animus():
    animus_arm = np.array([1.0, 2.0, 3.0])
    animus_none = np.array([0.5, 0.5, 0.5])
    result = _aff_from_arrays(animus_arm, animus_none, contact_arm=100.0, contact_none=100.0)
    assert abs(result.plateau - (2.0 - 0.5)) < 1e-9
    assert abs(result.per_contact - (2.0 - 0.5) / 100.0) < 1e-9


def test_aff_per_contact_uses_the_larger_denominator_and_never_divides_by_zero():
    animus_arm = np.array([1.0])
    animus_none = np.array([0.0])
    result = _aff_from_arrays(animus_arm, animus_none, contact_arm=0.0, contact_none=0.0)
    assert result.per_contact == 1.0  # denom floors at 1.0, not 0

    result2 = _aff_from_arrays(animus_arm, animus_none, contact_arm=10.0, contact_none=50.0)
    assert abs(result2.per_contact - 1.0 / 50.0) < 1e-9


def test_aff_ignores_nan_animus_entries():
    animus_arm = np.array([1.0, np.nan, 3.0])
    animus_none = np.array([0.0, 0.0, 0.0])
    result = _aff_from_arrays(animus_arm, animus_none, contact_arm=1.0, contact_none=1.0)
    assert abs(result.plateau - 2.0) < 1e-9  # nanmean([1,3]) == 2, not nan
