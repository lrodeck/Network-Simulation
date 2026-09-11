"""Experiment 03 infrastructure: the outcome-pair decomposition math (SS2),
tested on toy stance arrays rather than full runs -- fast, and isolates the
math from simulation noise. The scenarios are chosen to fail the way
Experiment 01's `affective_distance` failed (a raw signed mean of
un-oriented movement cancelling real, opposite-direction movement into a
false zero) if `_ideo_decomposition` regressed to that shape.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from discourse_lab.experiments.experiment03_bubble_intervention import (
    DeltaAff,
    DeltaIdeo,
    _aff_from_arrays,
    _cross_contact_from_frames,
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


# --------------------------------------------------------------------------
# V7.2: ideo_level_* retains the none arm's own absolute movement
# --------------------------------------------------------------------------


def test_ideo_level_columns_retain_the_none_arms_absolute_movement_when_delta_is_zero():
    """change-spec-v7-continuous-affect.md V7.2's own test: on a fixture
    where an arm's schedule is a no-op (here: the arm's post-period stance
    is IDENTICAL to none's -- the bit-identical-prefix property SS5.1's
    schedule fork guarantees for a genuine no-op arm), every differenced
    `toward_*`/`delta_k` component is exactly zero while `ideo_level_*` is
    not -- that non-zero number is what "toward_own_pole positive in
    113/120 cells" was actually reporting before V7.2 gave it a name of its
    own, separate from the intervention's OWN (here: absent) effect.
    """
    stance0 = _two_camps(-3.0, 3.0)
    stance1 = _two_camps(-2.0, 2.0)  # real background drift: both camps one unit closer to x=0

    result = _ideo_decomposition(stance0, stance1, stance1, k1_arm=2, k1_none=2)
    assert abs(result.toward_mean) < 1e-9
    assert abs(result.toward_other_camp) < 1e-9
    assert abs(result.toward_own_pole) < 1e-9
    assert result.delta_k == 0.0

    assert abs(result.ideo_level_toward_mean) > 0.5, "background drift should not read as zero"
    assert abs(result.ideo_level_toward_other_camp) > 0.5
    assert abs(result.ideo_level_toward_own_pole) > 0.5


def test_ideo_level_columns_are_nan_when_pre_period_camp_is_undefined():
    """Matching the existing NaN-gating test for `toward_*`: `ideo_level_*`
    is camp-relative too (it is `_movement`'s own output), so it must be NaN
    under the same unimodal-pre-period gate, not silently read as zero."""
    rng = np.random.default_rng(1)
    stance0 = rng.normal(0.0, 1.0, size=(40, 2))
    stance1 = rng.normal(0.0, 1.0, size=(40, 2))

    result = _ideo_decomposition(stance0, stance1, stance0, k1_arm=3, k1_none=1)
    assert np.isnan(result.ideo_level_toward_other_camp)
    assert np.isnan(result.ideo_level_toward_mean)
    assert np.isnan(result.ideo_level_toward_own_pole)


# --------------------------------------------------------------------------
# V7.4: cross-camp-restricted contact denominator
# --------------------------------------------------------------------------


def test_aff_per_cross_contact_reflects_the_cross_camp_restricted_denominator():
    """V7.4's own test: holding total engagement fixed while cross-camp
    share doubles, the total-volume denominator must leave `per_contact`
    unchanged and the cross-contact denominator must halve `per_cross_
    contact`."""
    animus_arm = np.array([2.0])
    animus_none = np.array([0.0])

    low_share = _aff_from_arrays(
        animus_arm, animus_none, contact_arm=100.0, contact_none=100.0,
        cross_contact_arm=10.0, cross_contact_none=10.0,
    )
    high_share = _aff_from_arrays(
        animus_arm, animus_none, contact_arm=100.0, contact_none=100.0,
        cross_contact_arm=20.0, cross_contact_none=20.0,
    )

    assert low_share.per_contact == pytest.approx(high_share.per_contact)
    assert high_share.per_cross_contact == pytest.approx(low_share.per_cross_contact / 2.0)


def test_aff_per_cross_contact_is_nan_without_the_join_inputs():
    result = _aff_from_arrays(np.array([1.0]), np.array([0.0]), contact_arm=10.0, contact_none=10.0)
    assert np.isnan(result.per_cross_contact)


def test_cross_contact_join_classifies_events_by_dyad_distance():
    """Pure-frame core of the engagement/author-stance join (V7.4's
    "engagement/author-stance join yielding per-event stance distance"):
    two engagements share a tick and a post but different engaging users --
    one close to the post's stance (same-camp-like), one far (cross-camp-
    like) -- and only the far one must classify as cross_contact.
    """
    engagements = pl.DataFrame({"t": [0, 0, 5], "user": [1, 2, 1], "post": [100, 100, 999]})
    traits = pl.DataFrame({
        "t": [0, 0, 5], "user": [1, 2, 1],
        "stance_0": [0.0, 5.0, 0.0], "stance_1": [0.0, 0.0, 0.0],
    })
    posts = pl.DataFrame({"post": [100, 999], "stance_0": [0.1, 0.1], "stance_1": [0.0, 0.0]})

    total, cross = _cross_contact_from_frames(
        engagements, traits, posts, ["stance_0", "stance_1"], d_cross=1.0, rms=False,
    )
    assert total == 3.0  # the raw engagement count, including the unjoined tick-5 row
    assert cross == 1.0  # only user 2 (distance ~4.9) exceeds d_cross=1.0


def test_cross_contact_join_rms_scales_distance_by_axis_count():
    """`rms=True` must divide by sqrt(D) before thresholding, matching
    `exposure/kernel.py::compute_features`'s own convention -- the same
    metric V7.3's mechanism uses, per V7.4's "so the two thresholds cannot
    drift apart"."""
    engagements = pl.DataFrame({"t": [0], "user": [1], "post": [100]})
    # raw Euclidean distance = 2.0 over D=4 axes; rms = 2.0 / sqrt(4) = 1.0,
    # exactly AT d_cross -- not "above" it, so this must NOT count as cross
    traits = pl.DataFrame({
        "t": [0], "user": [1],
        "stance_0": [2.0], "stance_1": [0.0], "stance_2": [0.0], "stance_3": [0.0],
    })
    posts = pl.DataFrame({
        "post": [100], "stance_0": [0.0], "stance_1": [0.0], "stance_2": [0.0], "stance_3": [0.0],
    })
    cols = ["stance_0", "stance_1", "stance_2", "stance_3"]

    _, cross_euclidean = _cross_contact_from_frames(engagements, traits, posts, cols, d_cross=1.0, rms=False)
    _, cross_rms = _cross_contact_from_frames(engagements, traits, posts, cols, d_cross=1.0, rms=True)
    assert cross_euclidean == 1.0   # raw distance 2.0 > 1.0
    assert cross_rms == 0.0         # rms distance 1.0 is not > 1.0


# --------------------------------------------------------------------------
# V7.5: BIC margin for emergent k
# --------------------------------------------------------------------------


def test_delta_bic_margin_is_never_nan_gated_unlike_the_camp_relative_components():
    rng = np.random.default_rng(1)
    stance0 = rng.normal(0.0, 1.0, size=(40, 2))  # unimodal pre-period: camp-relative components NaN
    stance1 = rng.normal(0.0, 1.0, size=(40, 2))

    result = _ideo_decomposition(
        stance0, stance1, stance0, k1_arm=3, k1_none=1,
        bic_margin_arm=0.4, bic_margin_none=0.9,
    )
    assert np.isnan(result.toward_mean)
    assert result.delta_bic_margin == pytest.approx(0.4 - 0.9)


def test_v7_5_bic_margin_decreases_monotonically_then_k_jumps():
    """change-spec-v7-continuous-affect.md V7.5's own test: a synthetic
    population morphed continuously from two clusters to three -- `bic_
    margin` must fall monotonically while k stays flat at 2, then k jumps
    to 3."""
    from discourse_lab.metrics.polarization import emergent_camps

    # Fixed within-cluster noise, drawn once, then shifted by `sep` -- the
    # only thing that varies across the sweep is the deterministic
    # separation, so a wobble cannot be re-sampling noise masquerading as
    # non-monotonicity.
    rng = np.random.default_rng(0)
    n = 150
    c0_noise = rng.normal(0.0, 0.5, (n, 1))
    c1_noise = rng.normal(0.0, 0.5, (n, 1))
    c2_noise = rng.normal(0.0, 0.5, (n, 1))

    ks, margins = [], []
    for sep in np.linspace(0.0, 6.0, 13):
        stance = np.concatenate([c0_noise - 4.0, c1_noise + 4.0, c2_noise + (-4.0 + sep)])
        result = emergent_camps(stance, k_max=5, seed=0)
        ks.append(result["k"])
        margins.append(result["bic_margin"])

    assert ks[0] == 2, f"a fully-merged third cluster should not register as its own k: {ks}"
    assert ks[-1] == 3, f"a well-separated third cluster should select k=3: {ks}"

    jump = next(i for i in range(1, len(ks)) if ks[i] != ks[i - 1])
    pre_jump = margins[:jump]
    assert all(a >= b - 1e-9 for a, b in zip(pre_jump, pre_jump[1:])), (
        f"bic_margin did not fall monotonically before the k jump at index {jump}: {pre_jump}"
    )
    assert ks[:jump] == [2] * jump, f"k moved before the margin-based jump point: {ks}"
