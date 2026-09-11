"""Experiment 03 -- When Is Popping a Filter Bubble Defensible?

A counterfactual on an already-polarized population: does an intervention
that increases cross-camp contact reduce affective hostility, deepen it, or
split ideological movement from affective movement entirely? Full design in
the uploaded brief ("Experiment 03 -- When Is Popping a Filter Bubble
Defensible?"); this module is a first pass at the infrastructure the full
brief needs. Wave A/A′ (SS5.4) have been run at reduced scale; Wave B
(SS5.3/5.4, named scenarios + LHS) and Wave C (SS5.2, hysteresis) have the
same driver infrastructure but are run selectively rather than at the
brief's own scale -- not the full multi-wave study.

Prerequisites this module assumes have landed (see FINDINGS.md):
  - Change spec V1-V6 (engagement valence, the de-escalation channel,
    endogenous valence, emergent k) -- without it two of the four possible
    answers to the experiment's question were unreachable by construction.
  - `dynamics.schedule` (config.py) -- SS5.1's "package prerequisite": a
    single run forks arms at a named tick with a bit-identical prefix,
    proved in tests/test_runner.py.
  - `graph.sbm_block_source="camp"` + `network.measures.cross_camp_tie_share`
    -- SS4's structural-tribalization dial, which did not previously exist
    as a continuous knob (archetype/topic_affinity SBM blocks are
    uncorrelated with camp; `homophily_beta` was already measured inert for
    it in experiments/intervention.py's own notes).
  - `population.animus_mu` / `population.stance_polarization` -- the
    affective and ideological dials SS4's table names. NEITHER existed
    before this module: animus's marginal was a hardcoded lognormal and
    stance axes were always plain normal(0,1). Built the same way as the
    structural dial, same session, same discipline (see PopulationConfig).

Deliberately scoped down from the full brief, each noted where it matters
and recorded in FINDINGS.md:
  - Wave A here is a ONE-FACTOR-AT-A-TIME sign screen, not a true Morris
    elementary-effects design (SS5.4 names "Morris screening"; a real Morris
    design needs multiple random trajectories through the 3-dial space to
    estimate global sensitivity, which is out of scope for a first pass).
  - No hysteresis phase (SS5.2), no response-surface fit, no calibration
    against a corpus (SS8), no viewpoint-diversity floor (SS2.3 -- a
    normative choice the brief itself defers). Wave A's own falsifier (H1:
    does the sign of Delta_aff flip anywhere in the swept space) does not
    need any of these.
  - Wave A itself covered a truncated ideological range: below the Sarle
    bimodality gate, `dynamics.affect_drive="camp"` (the mechanism Wave A
    ran under) froze the affect channel entirely, for every arm. V7.3
    (change-spec-v7-continuous-affect.md) replaces that with a continuous
    `"distance"` mechanism -- this module's own default since -- and V7.6
    re-gates the resulting substrate; `delta_aff.per_contact`'s
    total-engagement-volume normalization is joined by V7.4's
    cross-camp-restricted `per_cross_contact`, and `DeltaIdeo` gains
    `ideo_level_*` (V7.2) and `delta_bic_margin` (V7.5). Wave A′ -- the
    re-run this unlocks, over the full dial range -- has not been run
    (FINDINGS.md).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

import discourse_lab.exposure  # noqa: F401  (registers kernels/rankers)
from discourse_lab.analysis import set_param
from discourse_lab.config import Config
from discourse_lab.metrics import bimodality_coefficient
from discourse_lab.metrics.polarization import CAMP_BIMODAL_THRESHOLD, emergent_camps
from discourse_lab.runner import cached_run, load_run

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "experiment03"

# traits: the outcome pair needs per-user stance/animus snapshots (V7.4
# additionally joins these against engagements for per_cross_contact).
# engagements: delta_aff's per-contact denominator.
# posts: V7.4's join also needs each engaged post's own stance.
PERSIST = ("posts", "engagements", "traits")


# --------------------------------------------------------------------------
# Shared substrate + the 3 dials (SS4)
# --------------------------------------------------------------------------

def base_config(n_users: int, n_ticks: int) -> Config:
    """Everything every arm and every dial setting shares. `outrage`: the
    kernel this codebase already associates with polarization dynamics and
    whose outgroup weight is positive (FINDINGS.md), a reasonable read of
    "a platform whose existing engagement optimization already skews
    hostile" for an ALREADY-tribalized population. `valence_mode=
    "endogenous"` (V3): the feedback loop SS5.2's bistability question
    needs, and central to SS3's tautology-risk discussion (whether an
    already-hostile population metabolizes contact as attack).

    `affect_drive="distance"` (V7.3, change-spec-v7-continuous-affect.md):
    the whole point of this experiment is to measure the ideological dial's
    low end, which sits below the Sarle bimodality gate `"camp"` mode
    freezes the affect channel on (FINDINGS.md's "does popping the bubble
    help is not measurable" finding, from Wave A). Wave A′ (V7.6) is the
    re-run this unlocks.
    """
    cfg = Config()
    cfg = set_param(cfg, "population.n_users", n_users)
    cfg = set_param(cfg, "dynamics.n_ticks", n_ticks)
    cfg = set_param(cfg, "dynamics.kernel", "outrage")
    cfg = set_param(cfg, "population.affect", True)
    cfg = set_param(cfg, "dynamics.valence_mode", "endogenous")
    cfg = set_param(cfg, "dynamics.affect_drive", "distance")
    return cfg


DIAL_NAMES = ("affective", "ideological", "structural")


def dial_config(base: Config, *, affective: float, ideological: float, structural: float) -> Config:
    """The 3 dials (SS4), each in [0, 1]: 0 is the low/blind/unimodal end of
    the brief's own range for that dial, 1 is the high/sorted/bimodal end.
    Set independently -- this experiment's whole point is that they need
    not move together.

        affective:    population.animus_mu,          -2.2 (low)  .. 1.0 (high)
        ideological:  population.stance_polarization,  0.0 (uni) .. 8.0 (strongly bimodal)
        structural:   graph.sbm_homophily (camp blocks), 1.0 (blind) .. 0.05 (sorted)

    The ideological dial's Sarle-bimodality gate (metrics.polarization.
    CAMP_BIMODAL_THRESHOLD = 5/9) crosses around level ~0.5 at N~800-1000
    (calibrated empirically). BELOW the gate, camp is undefined, and under
    `dynamics.affect_drive="camp"` (the pre-V7.3 mechanism)
    `dynamics/drift.py::apply_drift`'s C1.3 affect op is gated on
    `camps is not None` -- so animus/identification do not update AT ALL,
    for ANY arm including `none`, not only the ENGAGEMENT/COMPOSITION arms
    whose `outgroup`/`outgroup_x_animus` kernel features are separately
    camp-conditional. Confirmed in Wave A (FINDINGS.md): every arm's
    delta_aff_plateau is EXACTLY 0.0 at ideological=0.1 (well below the gate
    at this module's mapping). That is a real, reportable model property
    under `"camp"` mode, not a bug: "does popping the bubble help" is not
    even MEASURABLE until the population has crossed into definable-camp
    territory, independent of whether popping it would help once there.
    `wave_a_screen`'s default background level was chosen to stay clear of
    this gate (see its docstring) for exactly that reason.

    V7.3 (change-spec-v7-continuous-affect.md) replaces this with
    `dynamics.affect_drive="distance"` -- `base_config`'s own default from
    V7.3 onward -- which has no bimodality gate at all: the affect channel
    is measurable at every ideological level, including 0.1. The paragraph
    above therefore describes `"camp"` mode (still reachable by overriding
    `affect_drive` back to it, e.g. to reproduce a pre-V7.3 result) rather
    than this module's own current behaviour; Wave A's own truncated-domain
    limitation is what V7.6's "Wave A′" re-run (FINDINGS.md, V7 sequencing)
    exists to lift.
    """
    for name, level in (("affective", affective), ("ideological", ideological), ("structural", structural)):
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"{name} dial level must be in [0, 1], got {level}")

    animus_mu = -2.2 + affective * (1.0 - (-2.2))
    stance_polarization = ideological * 8.0
    sbm_homophily = 1.0 - structural * 0.95

    return dataclasses.replace(
        base,
        population=dataclasses.replace(
            base.population, animus_mu=animus_mu, stance_polarization=stance_polarization,
        ),
        graph=dataclasses.replace(
            base.graph, generator="sbm", sbm_block_source="camp", sbm_homophily=sbm_homophily,
            # V7.6's re-gate (change-spec-v7-continuous-affect.md) is the
            # first time this substrate was ever run through `stylized_gate`
            # -- it failed reciprocity at 0.065 against the spec's 0.2-0.4
            # band with `sbm_mirror_p` left at its 0.0 default (an SBM-only
            # top-up `network/sbm.py::sbm_graph` needs and this module never
            # supplied; `graph.mirror_p`, the OTHER generators' knob, is not
            # read by `sbm_graph` at all). 0.15 was calibrated the same way
            # `experiments/gate.py::calibrated_gate_config` calibrated its
            # own `mirror_p` -- swept empirically against the measured
            # share, not the mirror probability itself (`network/
            # reciprocity.py`'s "mirror_p is NOT the reciprocity you then
            # measure") -- landing reciprocity at ~0.29-0.30, mid-band, at
            # both N=2,000 and N=10,000 (FINDINGS.md).
            sbm_mirror_p=0.15,
        ),
    )


# --------------------------------------------------------------------------
# The 4 arms (SS3), as `dynamics.schedule` overrides fired at the
# intervention tick
# --------------------------------------------------------------------------

# `kernel_theta` SET overrides, not `theta_scale`: a multiplicative scale on
# the `outgroup`/`cross_distance` feature flips sign depending on the base
# kernel's OWN convention (homophily's outgroup weight is negative,
# outrage's is positive -- exposure/kernel.py), so "scale it up" cannot
# reliably mean "more cross-camp engagement" across different burn-in
# kernels. A direct SET pins the intended direction regardless of the base
# kernel.
#
# Two tables, one per `targeting_mode` (experiment03-bubble-intervention.md
# §4, H3b): `"camp_pair"` overrides `outgroup` (binary, `camps[u] !=
# camps[author]`) -- matching how a platform would actually implement
# "recommend cross-camp creators" once it has assigned users to sides.
# `"distance"` overrides `cross_distance` instead (`phi(d(s_i,s_j))`,
# exposure/kernel.py -- the SAME saturating map V7.3 uses for the affect
# channel, camp-agnostic and unconditional).
#
# Wave A′ (FINDINGS.md) is why `"distance"` exists at all: `outgroup` only
# exists in `compute_features`'s output `if camps is not None`, so below
# the Sarle bimodality gate `"camp_pair"`'s override never fires and the
# `engagement` arm is a complete, bit-identical no-op vs `none` -- V7.3
# (change-spec-v7-continuous-affect.md) made the AFFECT channel's outcome
# measurable below that gate but did not touch this separate,
# camp-conditional consumer. `"distance"` targeting has no such gate.
ENGAGEMENT_KERNEL_THETA_CAMP_PAIR: tuple[tuple[str, str, float], ...] = (
    ("like", "outgroup", 0.8),
    ("reply", "outgroup", 1.2),
    ("quote", "outgroup", 1.0),
)
ENGAGEMENT_KERNEL_THETA_DISTANCE: tuple[tuple[str, str, float], ...] = (
    ("like", "cross_distance", 0.8),
    ("reply", "cross_distance", 1.2),
    ("quote", "cross_distance", 1.0),
)
TARGETING_MODES = ("camp_pair", "distance")


def _engagement_kernel_theta(targeting_mode: str) -> tuple[tuple[str, str, float], ...]:
    if targeting_mode == "camp_pair":
        return ENGAGEMENT_KERNEL_THETA_CAMP_PAIR
    if targeting_mode == "distance":
        return ENGAGEMENT_KERNEL_THETA_DISTANCE
    raise ValueError(f"unknown targeting_mode {targeting_mode!r}; expected one of {TARGETING_MODES}")


def arms_for(targeting_mode: str = "camp_pair") -> dict[str, tuple[tuple[str, object], ...]]:
    """The 4 arms (SS3), parameterized by `targeting_mode` (H3b). `"none"`
    is arm- AND targeting-mode-independent by construction (an empty
    overrides tuple), which is what lets `forked_config`'s withdrawal entry
    reuse it regardless of which mode an intervention arm ran under.
    """
    theta = _engagement_kernel_theta(targeting_mode)
    return {
        "none": (),
        "exposure": (("inject_k", 20),),
        "engagement": (("kernel_theta", theta),),
        # `valence_gamma0`, not `civility_prob`: `base_config` sets
        # `valence_mode="endogenous"` (V3), under which civility comes from
        # `dynamics/valence.py::assign_valence_endogenous`'s own logit
        # (P(civil) = sigma(gamma0 + gamma_animus*animus_i + gamma_dist*dist)),
        # and `civility_prob` is the EXOGENOUS (V1/V2) path's parameter --
        # simply unread here (caught by exactly this: composition and
        # engagement produced bit-identical smoke-test output before this fix).
        # Raising the intercept from its 0.0 default (P(civil)=0.5 at animus=0)
        # to 2.5 (sigma(2.5)=0.92) is the "friction on hostile replies /
        # moderation" reading of the composition arm's civility shift, applied
        # platform-wide rather than only to animus=0 users.
        "composition": (("kernel_theta", theta), ("valence_gamma0", 2.5)),
    }


# Module-level default (`"camp_pair"`, matching pre-targeting_mode
# behaviour byte-for-byte) so existing callers that never pass a targeting
# mode -- Wave A / Wave A′ among them -- keep their exact historical
# meaning. Prefer `arms_for(mode)` in new code.
ARMS: dict[str, tuple[tuple[str, object], ...]] = arms_for("camp_pair")
ARM_NAMES = ("none", "exposure", "engagement", "composition")
INTERVENTION_ARM_NAMES = ("exposure", "engagement", "composition")


def forked_config(
    burn_in: Config, arm: str, *, intervention_tick: int, withdrawal_tick: int | None = None,
    targeting_mode: str = "camp_pair",
) -> Config:
    """One config per arm, forked from `burn_in` via `dynamics.schedule` so
    every arm shares a bit-identical prefix through `intervention_tick`
    (SS5.1; proved in tests/test_runner.py's
    test_schedule_gives_a_bit_identical_prefix_and_diverges_after).
    `arms_for(...)["none"]` is `()`, an EMPTY overrides tuple --
    `effective_dynamics` treats "no schedule entry has fired yet" and "the
    fired entry changes nothing" identically (both resolve to the base
    dynamics, unmodified), which is exactly correct here since `burn_in`'s
    own field values ARE the none-arm's values by construction. A
    `withdrawal_tick` re-applies the none arm there too (SS5.2's hysteresis
    phase) -- targeting-mode-independent, since it is the same empty tuple
    under either mode.

    `targeting_mode` (H3b) picks which of the two `ENGAGEMENT_KERNEL_THETA_*`
    tables the `engagement`/`composition` arms use; see `arms_for`.
    """
    arms = arms_for(targeting_mode)
    if arm not in arms:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARM_NAMES}")
    schedule = [(intervention_tick, arms[arm])]
    if withdrawal_tick is not None:
        if withdrawal_tick <= intervention_tick:
            raise ValueError("withdrawal_tick must be after intervention_tick")
        schedule.append((withdrawal_tick, arms["none"]))
    return dataclasses.replace(
        burn_in, dynamics=dataclasses.replace(burn_in.dynamics, schedule=tuple(schedule))
    )


# --------------------------------------------------------------------------
# The outcome pair (SS2) -- always reported together, never collapsed to one
# number (Experiment 01's affective_distance lesson: a signed mean of raw,
# un-oriented coordinates can cancel real, opposite-direction movement into
# a false zero). Both delta_aff and delta_ideo are MODEL ARM minus NONE ARM
# -- the matched counterfactual SS5's fork design exists to produce, from
# the SAME burn-in and seed.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaAff:
    plateau: float       # mean animus at the post-intervention plateau, arm minus none
    per_contact: float   # the same delta, normalized by total engagement volume over the window
    # V7.4: the same delta normalized by CROSS-CAMP-RESTRICTED contact only
    # (events whose dyad stance distance exceeds `d_cross`) -- SS2.2's
    # literal ask, which a total-volume denominator cannot answer since the
    # arms deliberately change the cross-camp SHARE of engagement, not just
    # its volume. NaN when the join inputs (traits/posts persistence) are
    # unavailable, so `_aff_from_arrays` stays callable without them.
    per_cross_contact: float = float("nan")


@dataclass(frozen=True)
class DeltaIdeo:
    """Each of the first three is a per-user DISTANCE-DECREASE (or, for
    `toward_own_pole`, a camp-signed projection change) averaged across
    users -- oriented toward/away from its reference point BEFORE
    averaging, which is what keeps opposite-camp movement from cancelling
    the way a raw coordinate mean would. These three, and `ideo_level_*`
    below, are NaN whenever the PRE-intervention population is unimodal
    (camp is then a projection artifact, not two camps --
    metrics/polarization.py's own convention). `delta_k` and
    `delta_bic_margin` are never gated on that: V6(2)/V7.5 exist precisely
    so fragmentation is visible even where the binary camp frame is not.
    """
    toward_other_camp: float   # shrink in mean distance to the OTHER camp's pre-period centroid, arm minus none
    toward_mean: float         # shrink in mean distance to the pre-period global mean, arm minus none
    toward_own_pole: float     # growth in own-camp-signed movement along the pre-period dominant axis, arm minus none
    delta_k: float             # change in emergent k (metrics.polarization.emergent_camps), arm minus none
    # V7.5: the BIC-margin analogue of delta_k -- a gradient rather than a
    # step, so H5 is readable even where delta_k stays flat. Never NaN-gated.
    delta_bic_margin: float = float("nan")
    # V7.2: the `none` arm's OWN absolute movement (not net of anything) --
    # the background behaviour worth seeing precisely because it is NOT the
    # effect. `toward_*` above already differences it out; recovering an
    # arm's own absolute level, if ever needed, is `toward_* + ideo_level_*`.
    ideo_level_toward_other_camp: float = float("nan")
    ideo_level_toward_mean: float = float("nan")
    ideo_level_toward_own_pole: float = float("nan")
    # SS2.3's viewpoint-diversity floor `f`: post-intervention dispersion
    # over pre-intervention dispersion, for the arm's OWN trajectory (not
    # differenced against `none` -- `f` describes what the population ended
    # up looking like, not an effect size) and, for background context,
    # `none`'s own. Never NaN-gated: dispersion needs no camp split at all.
    # f < 1 is a diversity loss; f > 1 is a gain.
    diversity_ratio_arm: float = float("nan")
    diversity_ratio_none: float = float("nan")


def dispersion(stance: np.ndarray) -> float:
    """Total viewpoint spread across all stance axes (SS2.3): sum of
    per-axis variance -- equal to the trace of the covariance matrix, but
    without `np.cov`'s degenerate return shape at D=1. Camp-agnostic by
    construction: unlike bimodality (the distribution's SHAPE), this is
    purely about spread, so it needs no camp split to be well-defined.
    """
    return float(np.var(stance, axis=0).sum())


def diversity_ratio(stance0: np.ndarray, stance1: np.ndarray) -> float:
    """SS2.3's `f` for one run's own trajectory: `dispersion(stance1) /
    dispersion(stance0)`. NaN when the pre-period has zero dispersion (a
    degenerate fixture, never a real population)."""
    d0 = dispersion(stance0)
    if d0 <= 0:
        return float("nan")
    return dispersion(stance1) / d0


def _stance_and_animus_at(handle, cfg: Config, tick: int) -> tuple[np.ndarray, np.ndarray]:
    """This run's per-user (stance, animus) at the snapshot nearest `tick`."""
    from discourse_lab.population.traits import trait_names

    tr = handle.traits_used(cfg)
    at_t = tr.filter(pl.col("t") == tick).sort("user")
    if at_t.height == 0:
        raise ValueError(
            f"no trait snapshot at t={tick} (dynamics.snapshot_every="
            f"{cfg.dynamics.snapshot_every}; pick a tick on that cadence)"
        )
    names = trait_names(cfg)
    stance_cols = [n for n in names if n.startswith("stance_")]
    stance = at_t.select(stance_cols).to_numpy()
    animus = (
        at_t["animus"].to_numpy() if "animus" in at_t.columns else np.full(at_t.height, np.nan)
    )
    return stance, animus


def _fixed_axis(stance0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean, unit axis) of stance0's dominant principal component -- the
    same frame `metrics.stylized.stance_clusters` derives, kept explicit so
    a LATER stance array can be projected onto THIS frame instead of
    computing its own. Without that, "moved toward my own pole" would be
    scored against a moving target once the pole itself has drifted.
    """
    mean0 = stance0.mean(axis=0)
    centred = stance0 - mean0
    if centred.shape[1] == 1:
        return mean0, np.array([1.0])
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return mean0, vt[0]


def _aff_from_arrays(
    animus_arm: np.ndarray, animus_none: np.ndarray, contact_arm: float, contact_none: float,
    cross_contact_arm: float = float("nan"), cross_contact_none: float = float("nan"),
) -> DeltaAff:
    """Pure-array core of `delta_aff`, split out so the normalization logic
    is unit-testable on toy inputs without a persisted run. `cross_contact_*`
    (V7.4) are optional: omitted, `per_cross_contact` stays NaN rather than
    silently dividing by the wrong (total-volume) denominator."""
    plateau_delta = float(np.nanmean(animus_arm) - np.nanmean(animus_none))
    denom = max(float(contact_arm), float(contact_none), 1.0)
    per_contact = plateau_delta / denom

    per_cross_contact = float("nan")
    if np.isfinite(cross_contact_arm) and np.isfinite(cross_contact_none):
        cross_denom = max(float(cross_contact_arm), float(cross_contact_none), 1.0)
        per_cross_contact = plateau_delta / cross_denom

    return DeltaAff(plateau=plateau_delta, per_contact=per_contact, per_cross_contact=per_cross_contact)


def _cross_contact_from_frames(
    engagements: pl.DataFrame, traits: pl.DataFrame, posts: pl.DataFrame,
    stance_cols: list[str], d_cross: float, rms: bool,
) -> tuple[float, float]:
    """Pure-frame core of the V7.4 engagement/author-stance join: for each
    engagement (columns `t`, `user`, `post`), look up the ENGAGING user's own
    stance at that tick (`traits`: `t`, `user`, `stance_cols`) and the post's
    own stance (`posts`: `post`, `stance_cols`), then classify the event as
    cross-contact when their distance exceeds `d_cross` -- the SAME
    saturation midpoint V7.3's mechanism uses (`dynamics.affect_d0`), so the
    two thresholds cannot drift apart. Returns `(total_contact,
    cross_contact)`; `total_contact` is the RAW engagement count (matching
    `delta_aff`'s existing total-volume denominator) rather than the
    post-join count, so a snapshot cadence that ever missed a tick would
    undercount cross_contact specifically rather than silently also shrink
    the total-volume denominator it is meant to be compared against.
    """
    total_contact = float(engagements.height)
    if total_contact == 0:
        return 0.0, 0.0

    joined = engagements.join(traits, on=["t", "user"], how="inner")
    joined = joined.join(posts, on="post", how="inner", suffix="_post")
    if joined.height == 0:
        return total_contact, 0.0

    user_stance = joined.select(stance_cols).to_numpy()
    post_stance = joined.select([f"{c}_post" for c in stance_cols]).to_numpy()
    dist = np.linalg.norm(user_stance - post_stance, axis=1)
    if rms:
        dist = dist / np.sqrt(max(len(stance_cols), 1))

    cross_contact = float((dist > d_cross).sum())
    return total_contact, cross_contact


def _cross_contact_share(
    handle, cfg: Config, *, window_start: int, plateau_tick: int, d_cross: float
) -> tuple[float, float]:
    """`RunHandle`-reading wrapper around `_cross_contact_from_frames`."""
    engagements = handle.engagements().filter(
        (pl.col("t") >= window_start) & (pl.col("t") <= plateau_tick)
    ).select(["t", "user", "post"])

    # Positional `stance_{d}` names throughout, matching `posts.parquet`'s
    # own naming (io/store.py::posts_schema) exactly rather than
    # `trait_names(cfg)`'s (which would instead carry a scenario's named
    # axes, e.g. "stance_provision", were one ever loaded here) -- base_
    # config never loads a scenario, so the two naming conventions coincide,
    # but joining against `posts.parquet` needs the latter regardless.
    stance_cols = [f"stance_{d}" for d in range(cfg.stance_dims())]
    traits = handle.traits_used(cfg).select(["t", "user", *stance_cols])
    posts = handle.posts().select(["id", *stance_cols]).rename({"id": "post"})

    return _cross_contact_from_frames(
        engagements, traits, posts, stance_cols, d_cross, rms=(cfg.dynamics.agreement_metric == "rms"),
    )


def delta_aff(handle_arm, handle_none, cfg: Config, *, window_start: int, plateau_tick: int) -> DeltaAff:
    _, animus_arm = _stance_and_animus_at(handle_arm, cfg, plateau_tick)
    _, animus_none = _stance_and_animus_at(handle_none, cfg, plateau_tick)

    def _contact(handle) -> float:
        m = handle.metrics()
        window = m.filter((pl.col("t") >= window_start) & (pl.col("t") <= plateau_tick))
        return float(window["n_engagements"].sum())

    cross_arm, cross_none = float("nan"), float("nan")
    if handle_arm.has_traits and handle_none.has_traits and handle_arm.has_posts and handle_none.has_posts:
        d_cross = cfg.dynamics.affect_d0
        _, cross_arm = _cross_contact_share(
            handle_arm, cfg, window_start=window_start, plateau_tick=plateau_tick, d_cross=d_cross,
        )
        _, cross_none = _cross_contact_share(
            handle_none, cfg, window_start=window_start, plateau_tick=plateau_tick, d_cross=d_cross,
        )

    return _aff_from_arrays(
        animus_arm, animus_none, _contact(handle_arm), _contact(handle_none),
        cross_contact_arm=cross_arm, cross_contact_none=cross_none,
    )


def _ideo_decomposition(
    stance0: np.ndarray, stance1_arm: np.ndarray, stance1_none: np.ndarray, k1_arm: int, k1_none: int,
    bic_margin_arm: float = float("nan"), bic_margin_none: float = float("nan"),
) -> DeltaIdeo:
    """Pure-array core of `delta_ideo`, split out so the four-way
    decomposition is unit-testable on toy stance arrays. `stance0` is the
    SHARED pre-intervention state (SS5.1); `stance1_arm`/`stance1_none` are
    each run's own state at the plateau tick; `k1_arm`/`k1_none` (V7.5:
    `bic_margin_arm`/`bic_margin_none`) their emergent-k (and its BIC margin)
    at that same tick (computed by the caller, which already has
    `stance1_*` in hand — no reason to run k-means twice on the same array
    here).
    """
    mean0, axis = _fixed_axis(stance0)
    proj0 = (stance0 - mean0) @ axis
    bimodality0 = float(bimodality_coefficient(proj0))
    delta_k = float(k1_arm - k1_none)
    delta_bic_margin = float(bic_margin_arm - bic_margin_none)
    # SS2.3: needs no camp split at all, computed unconditionally up front
    # so it survives both return paths below.
    div_arm = diversity_ratio(stance0, stance1_arm)
    div_none = diversity_ratio(stance0, stance1_none)

    if not np.isfinite(bimodality0) or bimodality0 <= CAMP_BIMODAL_THRESHOLD:
        return DeltaIdeo(
            float("nan"), float("nan"), float("nan"), delta_k, delta_bic_margin=delta_bic_margin,
            diversity_ratio_arm=div_arm, diversity_ratio_none=div_none,
        )

    camp0 = (proj0 > np.median(proj0)).astype(np.int64)
    sign0 = np.where(camp0 == 1, 1.0, -1.0)
    centroid_1 = stance0[camp0 == 1].mean(axis=0)
    centroid_0 = stance0[camp0 == 0].mean(axis=0)
    other_centroid0 = np.where(camp0[:, None] == 1, centroid_0, centroid_1)

    def _movement(stance1: np.ndarray) -> np.ndarray:
        proj1 = (stance1 - mean0) @ axis
        toward_other = (
            np.linalg.norm(stance0 - other_centroid0, axis=1)
            - np.linalg.norm(stance1 - other_centroid0, axis=1)
        )
        toward_mean = np.linalg.norm(stance0 - mean0, axis=1) - np.linalg.norm(stance1 - mean0, axis=1)
        toward_pole = sign0 * (proj1 - proj0)
        return np.array([toward_other.mean(), toward_mean.mean(), toward_pole.mean()])

    # V7.2: `none`'s own absolute movement, retained (not just differenced
    # away) so the background drift `toward_own_pole`/`toward_mean` etc. are
    # measured against is itself visible -- it is what "toward_own_pole
    # positive in 113/120 cells" was actually reporting before this delta
    # existed to isolate the intervention's OWN contribution from it.
    level_none = _movement(stance1_none)
    net = _movement(stance1_arm) - level_none
    return DeltaIdeo(
        float(net[0]), float(net[1]), float(net[2]), delta_k,
        delta_bic_margin=delta_bic_margin,
        ideo_level_toward_other_camp=float(level_none[0]),
        ideo_level_toward_mean=float(level_none[1]),
        ideo_level_toward_own_pole=float(level_none[2]),
        diversity_ratio_arm=div_arm, diversity_ratio_none=div_none,
    )


def delta_ideo(handle_arm, handle_none, cfg: Config, *, pre_tick: int, plateau_tick: int) -> DeltaIdeo:
    # SS5.1's shared prefix means handle_none's own pre_tick snapshot is
    # bit-identical to handle_arm's -- reading it once, from either run, is
    # both correct and cheaper than reading (and asserting equal) twice.
    stance0, _ = _stance_and_animus_at(handle_arm, cfg, pre_tick)
    stance1_arm, _ = _stance_and_animus_at(handle_arm, cfg, plateau_tick)
    stance1_none, _ = _stance_and_animus_at(handle_none, cfg, plateau_tick)
    res_arm = emergent_camps(stance1_arm)
    res_none = emergent_camps(stance1_none)
    return _ideo_decomposition(
        stance0, stance1_arm, stance1_none, int(res_arm["k"]), int(res_none["k"]),
        bic_margin_arm=float(res_arm["bic_margin"]), bic_margin_none=float(res_none["bic_margin"]),
    )


def diversity_floor_break_even(delta_aff_plateau: float, diversity_ratio_arm: float) -> float:
    """SS2.3's normative rule, made a single reportable number instead of a
    verdict at one chosen floor: "An intervention is justified if it
    reduces affective hostility WITHOUT reducing viewpoint diversity below
    a floor f." Since `diversity_ratio_arm` is a single measured number,
    sweeping `f` needs no re-run -- the crossover is `diversity_ratio_arm`
    itself, for any arm that actually reduces hostility (`delta_aff_
    plateau < 0`): justified for every floor <= this value, not justified
    above it. An arm that does NOT reduce hostility is never justified
    regardless of floor, reported as NaN rather than a misleading number.
    """
    if not (delta_aff_plateau < 0.0):
        return float("nan")
    return diversity_ratio_arm


# --------------------------------------------------------------------------
# Running cells and Wave A (SS5.4)
# --------------------------------------------------------------------------

def run_arm(
    burn_in: Config, arm: str, seed: int, *, intervention_tick: int, n_ticks_total: int,
    targeting_mode: str = "camp_pair", withdrawal_tick: int | None = None,
):
    cfg = forked_config(
        burn_in, arm, intervention_tick=intervention_tick, targeting_mode=targeting_mode,
        withdrawal_tick=withdrawal_tick,
    )
    cfg = dataclasses.replace(cfg, dynamics=dataclasses.replace(cfg.dynamics, n_ticks=n_ticks_total))
    cached_run(cfg, seed, persist=PERSIST)
    return cfg, load_run(cfg, seed)


def run_design_point(
    burn_in: Config, seed: int, *, intervention_tick: int, n_ticks_total: int,
    targeting_mode: str = "camp_pair",
) -> list[dict]:
    """The `none` arm plus the 3 intervention arms at ONE (dial, seed)
    point, each of the 3 scored against `none` (SS2). `targeting_mode`
    (H3b) is recorded on every row so a Wave B sweep across both modes is
    distinguishable in the resulting table."""
    _, handle_none = run_arm(burn_in, "none", seed, intervention_tick=intervention_tick, n_ticks_total=n_ticks_total)
    pre_tick = intervention_tick - 1   # last tick still on the shared prefix
    plateau_tick = n_ticks_total - 1

    rows = []
    for arm in INTERVENTION_ARM_NAMES:
        arm_cfg, handle_arm = run_arm(
            burn_in, arm, seed, intervention_tick=intervention_tick, n_ticks_total=n_ticks_total,
            targeting_mode=targeting_mode,
        )
        aff = delta_aff(handle_arm, handle_none, arm_cfg, window_start=intervention_tick, plateau_tick=plateau_tick)
        ideo = delta_ideo(handle_arm, handle_none, arm_cfg, pre_tick=pre_tick, plateau_tick=plateau_tick)
        rows.append({
            "arm": arm, "seed": seed, "targeting_mode": targeting_mode,
            "delta_aff_plateau": aff.plateau, "delta_aff_per_contact": aff.per_contact,
            "delta_aff_per_cross_contact": aff.per_cross_contact,
            "toward_other_camp": ideo.toward_other_camp, "toward_mean": ideo.toward_mean,
            "toward_own_pole": ideo.toward_own_pole, "delta_k": ideo.delta_k,
            "delta_bic_margin": ideo.delta_bic_margin,
            "ideo_level_toward_other_camp": ideo.ideo_level_toward_other_camp,
            "ideo_level_toward_mean": ideo.ideo_level_toward_mean,
            "ideo_level_toward_own_pole": ideo.ideo_level_toward_own_pole,
            "diversity_ratio_arm": ideo.diversity_ratio_arm,
            "diversity_ratio_none": ideo.diversity_ratio_none,
            "diversity_floor_break_even": diversity_floor_break_even(aff.plateau, ideo.diversity_ratio_arm),
        })
    return rows


def wave_a_screen(
    n_users: int, n_ticks_burn_in: int, n_ticks_post: int, seeds: Sequence[int],
    dial_levels: Sequence[float] = (0.1, 0.5, 0.9), midpoint: float = 0.7,
) -> pl.DataFrame:
    """SS5.4 Wave A, scoped to what one session can run: a ONE-FACTOR-AT-A-
    TIME sign screen, not a true Morris elementary-effects design (module
    docstring). For each dial, sweep it across `dial_levels` holding the
    other two at `midpoint` -- the brief's own requirement is the SIGN of
    the effect, not the magnitude ("Wave A screens against the sign, not
    the magnitude").

    `midpoint` defaults to 0.7, not 0.5: `dial_config`'s docstring records
    that the ideological dial's bimodality gate crosses near level ~0.5, so
    a 0.5 background would hold the engagement/composition arms near a
    structural no-op (camp undefined) while sweeping the OTHER two dials --
    screening a degenerate slice of the space rather than the one the
    brief's H1-H3 are actually about. 0.7 keeps camp reliably defined as the
    shared background.
    """
    intervention_tick = n_ticks_burn_in
    n_ticks_total = n_ticks_burn_in + n_ticks_post
    rows = []
    for swept in DIAL_NAMES:
        for level in dial_levels:
            settings = {d: (level if d == swept else midpoint) for d in DIAL_NAMES}
            burn_in = dial_config(base_config(n_users, n_ticks_total), **settings)
            for seed in seeds:
                t0 = time.time()
                for row in run_design_point(
                    burn_in, seed, intervention_tick=intervention_tick, n_ticks_total=n_ticks_total
                ):
                    row.update({"swept_dial": swept, "level": level, **settings})
                    rows.append(row)
                print(f"  swept={swept} level={level} seed={seed}: {time.time() - t0:.1f}s")
    return pl.DataFrame(rows)


def smoke_test() -> None:
    # ideological=0.8 (not the [0,1] midpoint): dial_config's docstring
    # records the bimodality gate crossing near level ~0.5 at this scale, so
    # a smoke test at 0.5 would exercise only the camp-undefined branch of
    # delta_ideo/delta_aff and never touch the engagement/composition arms'
    # kernel_theta path at all.
    burn_in = dial_config(
        base_config(n_users=300, n_ticks=24), affective=0.6, ideological=0.8, structural=0.6
    )
    t0 = time.time()
    rows = run_design_point(burn_in, seed=0, intervention_tick=12, n_ticks_total=24)
    print(f"smoke: {time.time() - t0:.1f}s for 24 ticks, n_users=300")
    for row in rows:
        print(row)


def run_wave_a(
    n_users: int = 1000,
    n_ticks_burn_in: int = 60,
    n_ticks_post: int = 100,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> pl.DataFrame:
    """Defaults are the reduced-scale calibration recorded in FINDINGS.md:
    ~14s/design-point at N=800/80 ticks scaled to this size gave a full
    45-point sweep in a session-appropriate ~25 minutes. `n_ticks_post=100`
    (not 40): `lr_affect=0.015` is slow, and an early check at 40 post-
    intervention ticks showed deltas of order 1e-4 -- legible in sign but
    barely so; the longer window gives the plateau more room to separate
    from noise.
    """
    df = wave_a_screen(n_users, n_ticks_burn_in, n_ticks_post, seeds)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(RESULTS_DIR / "wave_a.csv")
    return df


def run_wave_a_prime(
    n_users: int = 1000,
    n_ticks_burn_in: int = 60,
    n_ticks_post: int = 100,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> pl.DataFrame:
    """Wave A′ (experiment03-bubble-intervention.md §5.4's "required" row;
    change-spec-v7-continuous-affect.md's own closing "Unlocks"): Wave A
    repeated at the SAME scale, under the now-default `dynamics.affect_
    drive="distance"`, over the FULL dial range.

    `midpoint=0.5`, not Wave A's 0.7: `wave_a_screen`'s 0.7 background
    existed only to keep the ideological dial's shared background clear of
    the `"camp"`-mode bimodality gate while sweeping the other two dials —
    a workaround for a defect `"distance"` mode does not have. 0.5 is the
    neutral midpoint of each [0, 1] dial, so this is a genuinely different
    slice of the space from Wave A's, not the same slice re-measured: it
    reaches ideological=0.1 with affective/structural ALSO at their
    neutral point, rather than the artificially-elevated 0.7 companion Wave
    A needed.

    Written to `wave_a_prime.csv`, never overwriting `wave_a.csv`: the two
    are not comparable on mechanism (`affect_drive`) OR structural grounds
    (`dial_config`'s `sbm_mirror_p=0.15`, added by the same V7.6 re-gate as
    the mechanism flip) OR background level, so treating this as "Wave A
    corrected in place" rather than a new, separate measurement would be
    the forking-paths mistake this brief's own inference discipline (§7)
    exists to prevent.
    """
    assert Config().dynamics.affect_drive == "distance", (
        "Wave A′'s whole point is measuring the distance-drive mechanism -- "
        "the config default must not have silently reverted to 'camp'"
    )
    df = wave_a_screen(n_users, n_ticks_burn_in, n_ticks_post, seeds, midpoint=0.5)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(RESULTS_DIR / "wave_a_prime.csv")
    return df


# --------------------------------------------------------------------------
# Wave B (SS5.3, SS5.4): named scenarios as strata, Latin Hypercube points
# inside each, both targeting modes, a response surface over delta_aff_plateau
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """One of SS5.3's "named worlds": a dial-level PRESET plus a short note
    on intent. SS5.3 wants scenarios defined by measurable properties a
    reader can locate on a real platform, not by dial settings -- the
    levels below are the KNOB, `describe_scenario` below measures the
    DEFINITION."""
    name: str
    affective: float
    ideological: float
    structural: float
    note: str


SCENARIOS: dict[str, Scenario] = {
    "consolidated_two_camp": Scenario(
        name="consolidated_two_camp", affective=0.7, ideological=0.9, structural=0.9,
        note=(
            "Strongly bimodal, structurally sorted, already fairly hostile "
            "-- the brief's 'two tight camps' world."
        ),
    ),
    "cross_cut": Scenario(
        name="cross_cut", affective=0.5, ideological=0.7, structural=0.1,
        note=(
            "Ideologically split but NOT structurally sorted -- ties cross "
            "camp lines freely even though positions do not. Deliberately "
            "NOT genuine multi-camp fragmentation (k>2): "
            "`population.stance_polarization` is a single bimodal axis, and "
            "Wave A/A' both measured delta_k at exactly 0.0 on every one of "
            "their 270 combined rows -- this codebase has no population "
            "generator that produces more than two ideological modes, so "
            "'fragmented multi-camp,' as such, is not currently buildable. "
            "This is the closest DISTINCT stratum the existing 3 dials can "
            "reach to that intent: tribalization that is ideological "
            "without being structural, rather than one splintered into more "
            "than two pieces."
        ),
    ),
    "low_tribalization": Scenario(
        name="low_tribalization", affective=0.1, ideological=0.1, structural=0.1,
        note="Near the origin on all three dials -- the region Wave A' (V7.3) made measurable for the first time.",
    ),
}


def describe_scenario(scenario: Scenario, n_users: int = 1000, seed: int = 0) -> dict:
    """SS5.3's own requirement: report the MEASURED properties a scenario's
    preset produces (population + graph only, no dynamics run, so this is
    cheap to call before committing to a scenario's LHS budget) -- so it is
    locatable by "cross-camp tie share 0.25, mean animus 2.1, two camps"
    rather than only by its dial levels.
    """
    from discourse_lab.metrics.polarization import camps_and_bimodality, emergent_camps
    from discourse_lab.network import cached_graph
    from discourse_lab.network.measures import cross_camp_tie_share
    from discourse_lab.population import cached_population
    from discourse_lab.runner import phase_rngs

    cfg = dial_config(
        base_config(n_users, 1), affective=scenario.affective,
        ideological=scenario.ideological, structural=scenario.structural,
    )
    rngs = phase_rngs(seed)
    pop = cached_population(cfg, seed, rngs["population"])
    graph = cached_graph(cfg, seed, pop, rngs["graph"])
    stance_cols = [i for i, n in enumerate(pop.trait_names) if n.startswith("stance_")]
    stance = pop.X_used[:, stance_cols]
    camps, bimodality = camps_and_bimodality(stance)
    k = int(emergent_camps(stance)["k"])
    labels = camps if camps is not None else (stance[:, 0] > np.median(stance[:, 0])).astype(np.int64)
    return {
        "scenario": scenario.name,
        "mean_animus": float(pop.animus.mean()) if pop.has_affect else float("nan"),
        "bimodality": bimodality,
        "camps_defined": camps is not None,
        "emergent_k": k,
        "cross_camp_tie_share": float(cross_camp_tie_share(graph.csr, labels)),
    }


def lhs_design_points(
    center: tuple[float, float, float], radius: float, n_points: int, seed: int,
) -> list[dict[str, float]]:
    """`n_points` Latin Hypercube samples in the 3-dial cube, confined to a
    `radius`-neighbourhood of `center` and clipped to [0, 1] (SS5.3: "~30
    Latin-hypercube points inside it" -- inside the named stratum, not
    spanning the whole space). Returns `{"affective", "ideological",
    "structural"}` dicts, one per point.
    """
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit = sampler.random(n=n_points)
    lo = [max(0.0, c - radius) for c in center]
    hi = [min(1.0, c + radius) for c in center]
    scaled = qmc.scale(unit, lo, hi)
    return [
        {"affective": float(a), "ideological": float(i), "structural": float(s)}
        for a, i, s in scaled
    ]


def run_wave_b(
    n_users: int = 1000,
    n_ticks_burn_in: int = 60,
    n_ticks_post: int = 100,
    n_lhs_points: int = 6,
    lhs_radius: float = 0.25,
    seeds: Sequence[int] = (0, 1),
    targeting_modes: Sequence[str] = TARGETING_MODES,
    scenarios: Sequence[str] = tuple(SCENARIOS),
) -> pl.DataFrame:
    """SS5.4 Wave B, reduced to what one session can run -- the SAME
    reduction Wave A applied to a "true Morris design" (module docstring),
    now applied to the brief's own Wave B: `n_lhs_points=6` per scenario
    (not ~30), `seeds=(0, 1)` (not 10), `lhs_radius=0.25` (a local
    neighbourhood around each named stratum, not the full space) -- roughly
    a 1/20 reduction in points per scenario, run over BOTH `targeting_mode`s
    (H3b) rather than the brief's implicit one, since that comparison is
    exactly what Wave A' surfaced a reason to make. At the brief's own full
    scale this is ~900 burn-ins / ~3,600 forks / ~10h.

    Each row is one (scenario, LHS point, targeting_mode, seed, arm) cell.
    `swept_dial`/`level` (Wave A's one-factor-at-a-time columns) do not
    apply -- LHS points vary all 3 dials at once -- so rows instead carry
    `lhs_index` (which of `n_lhs_points` within its scenario) alongside the
    actual `affective`/`ideological`/`structural` levels drawn, for
    grouping and for fitting a response surface after the fact.
    """
    intervention_tick = n_ticks_burn_in
    n_ticks_total = n_ticks_burn_in + n_ticks_post
    rows: list[dict] = []
    for scen_index, scen_name in enumerate(scenarios):
        scenario = SCENARIOS[scen_name]
        center = (scenario.affective, scenario.ideological, scenario.structural)
        # `100 + scen_index`, not `hash(scen_name)`: Python's string hash is
        # randomized per-process by default, which would make the LHS draw
        # (and so the whole sweep) non-reproducible run to run.
        points = lhs_design_points(center, lhs_radius, n_lhs_points, seed=100 + scen_index)
        for lhs_index, point in enumerate(points):
            burn_in = dial_config(base_config(n_users, n_ticks_total), **point)
            for targeting_mode in targeting_modes:
                for seed in seeds:
                    t0 = time.time()
                    for row in run_design_point(
                        burn_in, seed, intervention_tick=intervention_tick,
                        n_ticks_total=n_ticks_total, targeting_mode=targeting_mode,
                    ):
                        row.update({"scenario": scen_name, "lhs_index": lhs_index, **point})
                        rows.append(row)
                    print(
                        f"  scenario={scen_name} lhs={lhs_index} mode={targeting_mode} "
                        f"seed={seed}: {time.time() - t0:.1f}s"
                    )
    return pl.DataFrame(rows)


def run_wave_b_and_save(**kwargs) -> pl.DataFrame:
    df = run_wave_b(**kwargs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(RESULTS_DIR / "wave_b.csv")
    return df


# --------------------------------------------------------------------------
# Wave C (SS5.2): hysteresis -- withdraw the intervention, see what persists
# --------------------------------------------------------------------------

def _hysteresis_from_arrays(
    animus_arm_peak: np.ndarray, animus_none_peak: np.ndarray,
    animus_arm_final: np.ndarray, animus_none_final: np.ndarray,
) -> dict:
    """Pure-array core of `run_hysteresis`, split out so the recovery-
    fraction math is unit-testable on toy inputs without a persisted run
    (the same split `_aff_from_arrays`/`delta_aff` and `_ideo_decomposition`/
    `delta_ideo` already use).

    `peak_gap`/`final_gap` are the animus gap (arm minus none), each pair
    read at the SAME tick so both share whatever background drift `none`
    itself has at that point (V7.2's `ideo_level_*` discipline, applied
    here). `recovery_fraction = 1 - final_gap / peak_gap`: 1.0 is full
    recovery (gap fully closed), 0.0 is fully sticky (gap unchanged since
    withdrawal), negative is overshoot (the gap widened after withdrawal).
    NaN when `peak_gap` is exactly 0 -- there is nothing to recover from and
    nothing meaningful to divide by (this codebase has already found real
    arm/mode/regime combinations that are exact no-ops, e.g. `engagement`
    under `targeting_mode="camp_pair"` below the bimodality gate; dividing
    by that zero would report a fake, noise-dominated ratio instead of "not
    applicable").
    """
    peak_gap = float(np.nanmean(animus_arm_peak) - np.nanmean(animus_none_peak))
    final_gap = float(np.nanmean(animus_arm_final) - np.nanmean(animus_none_final))
    recovery_fraction = float("nan") if peak_gap == 0.0 else float(1.0 - final_gap / peak_gap)
    return {"peak_gap": peak_gap, "final_gap": final_gap, "recovery_fraction": recovery_fraction}


def run_hysteresis(
    burn_in: Config, arm: str, seed: int, *, intervention_tick: int, withdrawal_tick: int,
    n_ticks_total: int, targeting_mode: str = "camp_pair",
) -> dict:
    """H4 ("the hostile regime is stickier than the civil one"): does the
    animus gap an arm opened against `none` close again once the arm is
    switched back off? `forked_config` already supports this -- passing
    `withdrawal_tick` re-applies `arms_for(...)["none"]` (the empty-overrides
    no-op) at that tick, the SAME `dynamics.schedule` mechanism Wave A/B rely
    on for a bit-identical PREFIX, now used for a bit-identical post-
    withdrawal SUFFIX shape instead (arm and none converge on the same
    dynamics from there, just starting from different states).

    `RunHandle`-reading wrapper around `_hysteresis_from_arrays`: reads each
    run's animus at the tick just before withdrawal (`peak_tick`, the
    plateau the arm reached) and at the run's last tick (`final_tick`).
    """
    if not (intervention_tick < withdrawal_tick < n_ticks_total):
        raise ValueError("expected intervention_tick < withdrawal_tick < n_ticks_total")
    none_cfg, handle_none = run_arm(
        burn_in, "none", seed, intervention_tick=intervention_tick, n_ticks_total=n_ticks_total,
    )
    arm_cfg, handle_arm = run_arm(
        burn_in, arm, seed, intervention_tick=intervention_tick, n_ticks_total=n_ticks_total,
        targeting_mode=targeting_mode, withdrawal_tick=withdrawal_tick,
    )
    peak_tick = withdrawal_tick - 1
    final_tick = n_ticks_total - 1

    _, animus_arm_peak = _stance_and_animus_at(handle_arm, arm_cfg, peak_tick)
    _, animus_none_peak = _stance_and_animus_at(handle_none, none_cfg, peak_tick)
    _, animus_arm_final = _stance_and_animus_at(handle_arm, arm_cfg, final_tick)
    _, animus_none_final = _stance_and_animus_at(handle_none, none_cfg, final_tick)

    metrics = _hysteresis_from_arrays(
        animus_arm_peak, animus_none_peak, animus_arm_final, animus_none_final,
    )
    return {
        "arm": arm, "seed": seed, "targeting_mode": targeting_mode,
        "intervention_tick": intervention_tick, "withdrawal_tick": withdrawal_tick,
        "n_ticks_total": n_ticks_total,
        **metrics,
    }


def select_hysteresis_points(wave_b_df: pl.DataFrame, n_points: int = 3) -> pl.DataFrame:
    """SS5.2: 'Only run this where phase 2 produced a significant effect' --
    the `n_points` (scenario, lhs_index, targeting_mode, arm) cells with the
    largest seed-averaged |delta_aff_plateau| (averaging first so a single
    lucky seed cannot buy a slot). `arm="none"` is dropped: there is no
    intervention to withdraw from it.
    """
    keys = ["scenario", "lhs_index", "targeting_mode", "arm", "affective", "ideological", "structural"]
    return (
        wave_b_df
        .filter(pl.col("arm") != "none")
        .group_by(keys)
        .agg(pl.col("delta_aff_plateau").mean().alias("delta_aff_plateau_mean"))
        .sort(pl.col("delta_aff_plateau_mean").abs(), descending=True)
        .head(n_points)
    )


def run_wave_c(
    wave_b_df: pl.DataFrame,
    n_points: int = 3,
    n_users: int = 1000,
    n_ticks_burn_in: int = 60,
    n_ticks_pre_withdrawal: int = 60,
    n_ticks_post_withdrawal: int = 60,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> pl.DataFrame:
    """SS5.2's hysteresis phase, run ONLY on the `n_points` design points
    Wave B measured the largest |delta_aff_plateau| at (`select_hysteresis_
    points`) -- the brief's own scoping rule, and the reason this driver
    takes `wave_b_df` as an argument rather than a dial/arm/mode triple:
    which points qualify is an empirical question Wave B has to answer
    first, not a choice made here.
    """
    intervention_tick = n_ticks_burn_in
    withdrawal_tick = intervention_tick + n_ticks_pre_withdrawal
    n_ticks_total = withdrawal_tick + n_ticks_post_withdrawal
    points = select_hysteresis_points(wave_b_df, n_points)

    rows: list[dict] = []
    for point in points.iter_rows(named=True):
        burn_in = dial_config(
            base_config(n_users, n_ticks_total), affective=point["affective"],
            ideological=point["ideological"], structural=point["structural"],
        )
        for seed in seeds:
            t0 = time.time()
            row = run_hysteresis(
                burn_in, point["arm"], seed, intervention_tick=intervention_tick,
                withdrawal_tick=withdrawal_tick, n_ticks_total=n_ticks_total,
                targeting_mode=point["targeting_mode"],
            )
            row.update({
                "scenario": point["scenario"], "lhs_index": point["lhs_index"],
                "affective": point["affective"], "ideological": point["ideological"],
                "structural": point["structural"],
                "delta_aff_plateau_mean": point["delta_aff_plateau_mean"],
            })
            rows.append(row)
            print(
                f"  scenario={point['scenario']} arm={point['arm']} mode={point['targeting_mode']} "
                f"seed={seed}: {time.time() - t0:.1f}s"
            )
    return pl.DataFrame(rows)


def run_wave_c_and_save(wave_b_df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    df = run_wave_c(wave_b_df, **kwargs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(RESULTS_DIR / "wave_c.csv")
    return df


if __name__ == "__main__":
    import sys

    steps = sys.argv[1:] or ["smoke", "wave_a"]
    if "smoke" in steps:
        smoke_test()
    if "wave_a" in steps:
        run_wave_a()
    if "wave_a_prime" in steps:
        run_wave_a_prime()
    if "wave_b" in steps:
        run_wave_b_and_save()
    if "wave_c" in steps:
        run_wave_c_and_save(pl.read_csv(RESULTS_DIR / "wave_b.csv"))
