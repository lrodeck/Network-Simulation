"""Nested frozen configs with structural hashing (dev notes §5, §8.2).

Every sub-config hashes independently and canonically (sorted keys, fixed float
formatting), so artifact keys are derived from exactly the config that
determines the artifact. A run is fully described by JSON.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# canonical serialisation
# --------------------------------------------------------------------------

def _canonical(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("config contains non-finite float")
        return round(obj, 12)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 12)
    if isinstance(obj, np.ndarray):
        return [_canonical(x) for x in obj.tolist()]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _canonical(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if not f.name.startswith("_")
        }
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    raise TypeError(f"cannot canonicalise {type(obj)}")


def canonical_json(obj: Any) -> str:
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"))


def structural_hash(obj: Any) -> str:
    return hashlib.blake2b(canonical_json(obj).encode("utf-8"), digest_size=16).hexdigest()


class Hashable:
    """Mixin giving every sub-config an independent structural hash."""

    def hash(self) -> str:
        return structural_hash(self)

    def to_dict(self) -> dict:
        return _canonical(self)

    def to_json(self) -> str:
        return canonical_json(self)


# --------------------------------------------------------------------------
# sub-configs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PopulationConfig(Hashable):
    n_users: int = 10_000
    n_topics: int = 8
    # spec §1.1: "D ~ 3-5 latent ideological axes"; §4.3's config sketch says
    # 3. A scenario, when loaded, overrides this with its own axis count.
    # This was -1, and Config.stance_dims() floors at 1, so every run without
    # a scenario silently collapsed stance to a single axis — which is not a
    # smaller version of the model but a different geometry: with D=1 there is
    # no orientation for homophily to be homophilous *in*, and §7.5's
    # orthogonal-vs-correlated axes question cannot be posed at all.
    stance_dims: int = 3
    archetype_weights: tuple[tuple[str, float], ...] = ()   # () → library defaults
    archetype_offsets: tuple[tuple[str, str, float], ...] = ()  # (archetype, trait, offset)
    # (trait_i, trait_j, rho). ADDS to whatever the archetype mixture already
    # induces — it does not set the realised correlation. An archetype that
    # shifts two traits together correlates them, and the two mechanisms
    # compose without either knowing about the other. Measured at N=20000 on
    # the shipped defaults, for activity x reply_prop:
    #
    #     archetypes off, no pairs           -0.001
    #     archetypes off, asked for 0.30     +0.299   <- you get what you ask
    #     archetypes on,  no pairs           +0.303   <- the lurker archetype
    #     archetypes on,  asked for 0.30     +0.493   <- both, added
    #
    # `sample_population` warns when a requested pair touches traits an
    # archetype also moves. To control a pair exactly, use one mechanism or
    # the other.
    #
    # NOT defaulted to anything: an empty tuple gives
    # an identity correlation matrix, so every trait — including the stance
    # axes — is independent unless listed here. spec §7.5 notes that correlated
    # stance axes are what produce the empirically observed collapse toward a
    # single dominant dimension, and measured here they do: cross-camp exposure
    # falls from 0.362 to 0.326 at rho=0.85. Trait names must match this
    # config's own columns (see semantics.Lexicon.trait_column).
    correlation_pairs: tuple[tuple[str, str, float], ...] = ()
    # Gini of a lognormal is erf(sigma/2) in closed form, so this parameter
    # *is* the spec §5.1 posting-volume inequality target. The spec's own
    # sigma = 1.2 gives 0.604 against its stated target of 0.7-0.9 — the two
    # clauses are mutually incompatible. 1.8 gives 0.797, mid-band, with the
    # top 1% of users producing ~30% of posts.
    activity_sigma: float = 1.8
    pareto_alpha: float = 2.3
    topic_logit_sigma: float = 1.0
    # C1 (change spec): the affect block — `identification` (attachment to
    # own camp, logit) and `animus` (hostility toward the opposing camp,
    # log). Affective polarization is the outcome variable half the relevant
    # literature measures (Iyengar, Sood & Lelkes 2012; Iyengar & Westwood
    # 2015; Törnberg, PNAS 2022; Rathje et al. 2021), and ideological
    # position alone cannot represent it. This is a *population-structural*
    # switch rather than a dynamics knob on purpose: it changes the trait
    # layout, so it must fork the population artifact key — a cached
    # population with a different column count silently reused would be the
    # worst kind of bug a content-addressed cache can hide.
    affect: bool = False
    # Experiment 03 §4's affective-tribalization dial: the log-space MEAN of
    # the initial `animus` draw (population/traits.py::_affect_marginal's
    # lognormal). -2.2 is the pre-existing shipped default (low mean, heavy
    # right tail); raising it moves the whole population's starting
    # hostility up, independent of `stance_polarization` and
    # `graph.sbm_homophily` below. Only read when `affect=True`.
    animus_mu: float = -2.2
    # Experiment 03 §4's ideological-tribalization dial: 0 leaves stance
    # axis 0 at the plain `normal(0,1)` every other axis uses (unimodal —
    # today's only behaviour); above 0 it draws from `bimodal_normal`
    # instead (population/marginals.py), an equal-weight two-Gaussian
    # mixture at +/-`stance_polarization`/2 — a continuous unimodal ->
    # strongly-bimodal dial at fixed n_users/archetype mix, independent of
    # the other two dials. Axis 0 only: `camps_and_bimodality` projects onto
    # the dominant component, and axis 0's variance dominates as soon as
    # separation makes it the largest, so one axis is enough to move camp
    # bimodality without also changing every other axis's marginal shape.
    # Ignored once a scenario supplies its own stance_axes marginals.
    stance_polarization: float = 0.0


@dataclass(frozen=True)
class GraphConfig(Hashable):
    generator: str = "latent_space"           # latent_space | latent_pa | sbm | configuration | barabasi
    mean_degree: float = 40.0
    homophily_beta: float = 0.35              # β on latent distance
    prominence_gamma: float = 0.6             # γ on log(1 + prominence)
    # Probability of mirroring each generated edge — NOT the measured
    # reciprocity of the result, which is what spec §5.1's 0.2-0.4 target
    # refers to. Mirroring a fraction r yields 2r/(1+r) reciprocated edges,
    # on top of a ~0.16 baseline the homophilous generator produces by
    # chance. 0.10 measures ~0.30.
    mirror_p: float = 0.02
    fanout_cap: int = 400                     # max followers reached per post per tick
    knn_k: int = 60                           # candidate pool when N is large
    long_tie_fraction: float = 0.1            # uniform random component in kNN graphs
    # `latent_pa` only: share of edges drawn globally with destination
    # probability proportional to prominence, rather than from the kNN pool.
    # The dial between local structure (clustering) and a heavy in-degree
    # tail (spec §5.1's engagement rows). Ignored by every other generator.
    pa_fraction: float = 0.35
    sbm_blocks: int = 0                       # 0 → derive from sbm_block_source instead
    sbm_block_source: str = "archetype"       # archetype | topic_affinity | camp (used when sbm_blocks == 0)
    sbm_homophily: float = 0.8
    # `mirror_p` (below) is calibrated against the `latent_pa` generator's
    # naturally denser structure; the SBM's independent per-ordered-pair draws
    # give it a baseline reciprocity of ~p_within (2-3% at typical block
    # counts/mean_degree), so the shared mirror_p alone leaves it well under
    # the spec §5.1 band. This is an SBM-only top-up, applied inside
    # sbm_graph() in addition to the shared post-pass.
    sbm_mirror_p: float = 0.0

    def __post_init__(self) -> None:
        # The kNN pool is the set of candidates homophily_beta and
        # prominence_gamma then *weight*. If the pool is no bigger than the
        # degree being drawn from it, every candidate is taken and both
        # weights become inert — the generator silently degrades to plain
        # kNN. Measured: at knn_k=40, mean_degree=40, sweeping beta from
        # 0.35 to 1.5 changed clustering by exactly nothing.
        if self.knn_k <= self.mean_degree:
            raise ValueError(
                f"knn_k={self.knn_k} <= mean_degree={self.mean_degree}: the candidate pool "
                "leaves no room for homophily_beta or prominence_gamma to select, so both "
                "become inert. Raise knn_k above mean_degree."
            )


@dataclass(frozen=True)
class DynamicsConfig(Hashable):
    n_ticks: int = 500
    posts_per_tick_rate: float = 0.02         # Poisson rate at activity = 1
    ticks_per_day: int = 24
    fatigue_decay: float = 0.9

    # b in B_u ~ Poisson(b · activity). NOTE: composes with tau_position, and at
    # this default the position decay binds first — the budget removes ~1% of
    # what decay already let through. See exposure/attention.py before sweeping it.
    attention_budget: float = 30.0
    tau_position: float = 6.0                 # position decay exp(-r / tau)
    inject_k: int = 0                         # algorithmic injections per post
    ranker: str = "chronological"
    kernel: str = "homophily"
    kernel_theta: tuple[tuple[str, str, float], ...] = ()   # (feature, action, value)
    # C3b's named coefficient groups, second consumer: a POPULATION-level
    # multiplier on a group's coefficients — theta_u = theta_base * g_u *
    # theta_scale[group]. Per-user *gains* (kernel_learning) vary the theory
    # across people; this varies the theory itself, which is what makes a
    # ladder like "social_proof strength 0 → 4" expressible without swapping
    # whole kernels (and so without losing the crossing point D3 exists to
    # find). (group, value) pairs; groups are those in exposure/kernel.py's
    # FEATURE_GROUPS — features outside any group (quality, novelty…) cannot
    # be scaled this way.
    theta_scale: tuple[tuple[str, float], ...] = ()
    # How stance disagreement is measured in the engagement kernel (spec §2.6
    # `phi`), and therefore whether a theta authored at one stance
    # dimensionality means the same thing at another. spec §7.5 leaves the
    # treatment of multiple stance axes open; this is that choice.
    #
    #   "euclidean"  ||s_u - s_p||, the total distance. Its mean grows like
    #                sqrt(D) (-1.13 at D=1, -2.26 at D=3, -3.01 at D=5) while
    #                its spread barely moves (0.86 -> 0.95 -> 0.97), so raising
    #                D silently subtracts a constant from every utility and
    #                suppresses engagement — a D-dependent intercept shift
    #                wearing a feature's clothes.
    #   "rms"        the same distance per dimension, ||s_u - s_p|| / sqrt(D),
    #                so the feature has the same location and scale at any D
    #                and a kernel theta transfers across dimensionalities.
    agreement_metric: str = "rms"

    # C13 recalibration: with reply_selection="kernel" the Hawkes draw is
    # gated on candidate availability (a thread with no kernel-reply
    # candidates draws on excitation alone), and the higher reply-action
    # rate spreads attention more evenly — mu0 0.004 -> 0.01 and the C9
    # theta_scale recalibrated together to recenter both §5.1 rows. 20-seed
    # gates in test_change_spec (C9 and C13) pin the combination.
    hawkes_mu0: float = 0.01                  # baseline reply intensity per tick
    hawkes_ratio: float = 0.6                 # alpha/beta, must stay < 1
    hawkes_beta: float = 1.5
    max_thread_age: int = 15                  # ticks a thread stays open for Hawkes
    # Fraction of its parent's current reply intensity that a new reply post's
    # own thread opens with. spec §2.3 gives lambda_p per post and leaves mu_p
    # unspecified, so this is the §7-style open choice made into a dial.
    #
    #   0.0  every post opens at hawkes_mu0 — the literal reading. A reply
    #        inside a raging thread is as cold as a fresh post, so depth
    #        cannot compound and thread depth sits at ~1.06.
    #   >0   heat propagates down the chain: replying to a hot reply is
    #        itself likely, which is what makes threads deep rather than wide.
    #
    # Values above 1 are meaningful and are where the useful regime is: a
    # reply lands in a conversation already hotter than a cold post, so its
    # own thread starts hotter still. What bounds it is stability, not 1.
    #
    # Measured (n_users=1500, 40 ticks; stability over 90 ticks at n=800),
    # with max_replies_per_tick = 1:
    #
    #   inherit   P(branch|root)  P(branch|in-thread)  singleton  depth
    #   0.6                0.085                0.123      0.915   1.16
    #   1.8                0.092                0.294      0.908   1.43
    #   1.0                    -                    -          -   ~1.2   <- default
    #   1.8                0.092                0.294      0.908   1.43
    #   2.65               0.080                0.55       0.926   2.30
    #   3.0                0.080                0.608      0.920   2.54
    #
    # The default is 1.0, not the depth-optimal 2.65, because deep threads
    # dilute the thing Experiment 1 measures. Hawkes replies are not
    # kernel-driven — a reply carries the replier's own stance — so the more
    # of the corpus is replies, the less of what a user consumes was selected
    # by the engagement kernel, and the §5.3 null comparison loses power.
    # Measured at n_users=800, n_ticks=20, 20 seeds, D=1, the homophily
    # agreement effect against its matched null:
    #
    #   inherit 0.6 -> t=+3.77    1.0 -> t=+2.46    1.8 -> t=+1.93
    #   inherit 2.65 -> t=+0.74 (indistinguishable from noise)
    #
    # spec §5.1's depth row is a description of the model; §5.2/§5.3 are what
    # the model is for. Depth 1.5-3 is reachable and stable at inherit >= 2.2
    # and is an experimental condition to select deliberately, not the
    # default. Setting it costs the null comparison its resolution.
    #
    # That separation is what spec §5.1 actually requires: its two cascade
    # rows (>90% singletons AND depth 1.5-3) can only both hold if roots
    # branch rarely while threads already started continue often. A flat
    # branching probability gives depth = 1/(1-p), which is 1.1 at p = 0.09
    # no matter how the other knobs are set.
    #
    # Uncapped (max_replies_per_tick = 0) every depth-productive setting was
    # supercritical: at inherit 1.8, replies/tick went 6 -> 4453 by tick 30;
    # at 2.5 the run exhausted memory. The tick warns when replies run away.
    hawkes_mu_inherit: float = 1.0
    # Arrivals per post per tick; 0 = uncapped Poisson. At 1 a conversation
    # extends as a chain rather than a bush, which is what gives depth without
    # runaway volume — see HawkesThreads.step.
    max_replies_per_tick: int = 1

    trend_eta: float = 0.3                    # topic susceptibility to discourse state
    post_lifetime: int = 5                    # ticks a post stays in candidate inboxes
    rho_s: float = 0.9                        # discourse attention decay
    rho_sigma: float = 0.9                    # dominant stance decay
    cascade_depth_decay: float = 0.7          # rho^depth visibility
    max_cascade_depth: int = 25
    max_cascade_size: int = 1000              # warning threshold, per tick

    drift: str = "full"                       # none | social | full
    drift_lr: float = 0.02                    # reinforcement channel
    drift_lr_social: float = 0.01             # social influence channel
    # C3a (change spec): extends channel 1's gradient to the behavior block
    # (activity, reply_prop, repost_prop — the columns with a generation-map
    # path). Brady, McLoughlin, Doan & Crockett (Science Advances 2021) show
    # outrage expression is socially *learned*: positive feedback raises
    # future outrage. A kernel that encodes outrage as a fixed disposition
    # cannot produce norm convergence.
    drift_lr_behavior: float = 0.01
    drift_ramp_ticks: int = 50                # gains ramp linearly from 0 over this many ticks
    ou_k: tuple[tuple[str, float], ...] = ()  # (block, rate) overrides
    noise_sigma: float = 0.002
    llm_adjudication: bool = False            # queued only; offline pass in v1

    # -- C1: affect dynamics -------------------------------------------------
    # Step size of the affect_update drift op. Affect is a *state*, not a
    # fixed trait: it updates from interaction outcomes (C1.3), which is why
    # it is a drift channel and not just two new columns.
    lr_affect: float = 0.015
    # Mean-reversion for the affect block. 0.02 sits between expression's
    # 0.05 (style is fashion) and stance's 0.005 (position is sticky):
    # affect is stickier than style, less sticky than position. This is a
    # guess, not a calibrated value — flagged in MODEL.md §15 and carried in
    # the C10 sensitivity sweep.
    affect_ou_k: float = 0.02
    # C1.3's weight tables, config-side for the same reason C6 moved channel
    # 2's. `skip` is structurally 0 and not a dial. (action, weight) pairs;
    # unlisted actions are 0.
    #
    # V2 (change spec V1-V6, "Engagement Valence and the De-escalation
    # Channel"): re-keyed on (action, valence). This table now supplies only
    # the per-ACTION MAGNITUDE (a reply weighs more than a like) — every
    # entry is >= 0 by construction. The SIGN comes from `affect_valence_signs`
    # below: `weight(action) * valence_sign_and_magnitude(cell)`. `report` is
    # REMOVED from this table entirely — it is disengagement (the user hands
    # the conflict to the platform and exits), not a hostility increment, and
    # V4 makes it an exit event acting on future exposure instead.
    affect_weights_hostility: tuple[tuple[str, float], ...] = (
        ("like", 0.25), ("repost", 0.25), ("quote", 0.5), ("reply", 0.5),
    )
    affect_weights_support: tuple[tuple[str, float], ...] = (
        ("like", 1.0), ("repost", 1.5), ("quote", 0.5),
        ("reply", 0.5), ("report", -1.0),
    )
    # V2: sign fixed by theory, magnitude free (like `affect_ou_k`, carried
    # in the C10 sensitivity sweep rather than calibrated). Keys are
    # "{agree,disagree}_{civil,hostile}".
    #   disagree_hostile  +  the backfire channel — retains pre-V2 behaviour
    #   disagree_civil    -  Allport's contact hypothesis, operationalized:
    #                        the de-escalation channel V1-V4 exists to add
    #   agree_civil       -  affirmation; small because it is mostly in-group
    #   agree_hostile     +  the pile-on cell (in-group bonding against an
    #                        out-group) — not made smaller than
    #                        disagree_hostile, since V1's own warrant is that
    #                        this cell carries social reward and is the most
    #                        likely to dominate
    affect_valence_signs: tuple[tuple[str, float], ...] = (
        ("disagree_hostile", 1.0), ("disagree_civil", -1.0),
        ("agree_civil", -0.3), ("agree_hostile", 1.2),
    )
    # V7.3 (continuous affect drive): the affect op's drive term. "camp" is
    # the pre-V7.3 mechanism, byte-identical to before — keyed on the binary
    # in/out-group label from `camps_and_bimodality`, which is undefined (and
    # so gates the whole op off) below the Sarle bimodality threshold.
    # "distance" (the default since V7.6's re-gate passed — FINDINGS.md)
    # replaces the binary with a saturating function of continuous per-axis
    # stance distance (see `dynamics/drift.py::affect_delta`) and never gates
    # on bimodality, so a population that has not yet sorted into two camps
    # still has a measurable affect channel. Set back to "camp" to reproduce
    # a pre-V7.3 result (e.g. Experiment 01's SBM finding) under the exact
    # mechanism it was measured with.
    affect_drive: str = "distance"      # camp | distance
    # V7.3: the saturation constant in phi(d) = d / (d + affect_d0). Fixed at
    # the (approximate) median pairwise stance distance of a standard-normal
    # one-axis population, not calibrated per-run like `agree_delta` — a
    # deliberately simple constant, carried in the C10-style sensitivity
    # sweep rather than tuned. Also V7.4's `d_cross` threshold, so the two
    # cannot drift apart.
    affect_d0: float = 1.0

    # V1: the two valence axes assigned at the moment of engagement.
    # agree/disagree is derived from the kernel's own `agreement` feature
    # (thresholded against each tick's own median, so the split needs no
    # calibrated distance and is stable across stance dimensionality); civil
    # /hostile is exogenous here — a coin flip at `civility_prob` — until V3
    # replaces it with a per-user logit on animus and stance distance.
    # `force_agree` / `force_civil` pin an axis for EVERY engagement this
    # run: the fixture hook change-spec V1's own test needs ("every
    # engagement is agree+civil" / "...disagree+hostile"), not a production
    # dial.
    civility_prob: float = 0.5
    force_agree: bool | None = None
    force_civil: bool | None = None

    # -- V3: endogenous valence ------------------------------------------------
    # "exogenous" (V1/V2, default): civility is `civility_prob`, agree/disagree
    # a fixed geometric threshold. "endogenous": both come from the engaging
    # user's own state and the dyad's geometry (dynamics/valence.py::
    # assign_valence_endogenous) -- the feedback loop the bistability question
    # needs (does a population already in a hostile regime metabolize added
    # contact as attack). Requires `population.affect` (P(civil) reads the
    # user's own animus). None of the eight coefficients below are empirically
    # anchored (same status as `affect_ou_k`); signs are fixed by theory
    # (see EndogenousValenceParams), magnitudes are swept.
    valence_mode: str = "exogenous"       # exogenous | endogenous
    valence_beta0: float = 0.0
    valence_beta_dist: float = -1.0
    valence_beta_ident: float = -0.3
    valence_noise_agree: float = 1.0
    valence_gamma0: float = 0.0
    valence_gamma_animus: float = -1.0
    valence_gamma_dist: float = 0.0
    valence_noise_civil: float = 1.0

    # -- V4: report as an exit event ------------------------------------------
    # `report` moved out of the hostility table (V2); this is what replaces
    # it. `report_exit=True` makes a report suppress the reporter's future
    # exposure to that author (dynamics/report_exit.py) — defaulted off like
    # every other change-spec mechanism until asked for.
    # `report_animus_increment` is a free parameter (spec: "a free parameter,
    # defaulting to zero") for a small direct animus effect from the act
    # itself, applied outside the (action, valence) hostility table.
    report_exit: bool = False
    report_animus_increment: float = 0.0

    # -- C2: selection layer and tie rewiring ---------------------------------
    # Bakshy, Messing & Adamic (2015) found individual choice filtered
    # cross-cutting content more than the algorithm did. The selection stage
    # sits between exposure and reaction; `position_only` preserves current
    # behaviour exactly.
    selection: str = "position_only"          # position_only | homophilous | arousal_seeking
    # (name, value) overrides for the selection logits' betas: beta_pos,
    # beta_agree, beta_arousal.
    selection_beta: tuple[tuple[str, float], ...] = ()
    # Slow follow/unfollow process (C2.2): unfollow on accumulated hostile
    # interaction, follow on accumulated positive engagement. Runs every
    # `rewire_every` ticks — a per-tick sparse rebuild at N=1e4 is the one
    # thing here that could plausibly dominate runtime.
    rewire: bool = False
    rewire_every: int = 25
    rewire_rate: float = 0.01

    # -- C3b: learnable kernels ------------------------------------------------
    # Three tiers (change spec C3b): "none" keeps theta a readable table;
    # "group_gain" learns a per-user gain over named coefficient groups
    # (theta_u = theta_base * g_u); "full" is unrestricted per-user theta,
    # shipped for honesty and expected to be rarely used. Learning modulates
    # a named kernel, never replaces it — a run must stay describable as
    # "outrage, plus this much learned deviation". The learning rule is named
    # in the config, never implied: conformity (Brady's norm convergence),
    # bandit (own-reinforcement), habituation (mere exposure; the control).
    kernel_learning: str = "none"             # none | group_gain | full
    kernel_learning_rule: str = "conformity"  # conformity | bandit | habituation
    lr_kernel: float = 0.01
    kernel_gain_ou_k: float = 0.02            # gain reversion toward g = 1

    # -- C4: the quality backdoor ----------------------------------------------
    # quality is generated from author traits, and those same traits drive
    # prominence and activity — so Spearman(quality, engagement) is nonzero
    # even under the null kernel and measures author-trait alignment, not
    # merit. 0.0 draws quality author-trait-independent (the Salganik/
    # Muchnik condition Experiment 3 needs); 1.0 is the old behaviour.
    quality_trait_coupling: float = 0.0

    # -- C6: repulsion as an ablatable switch -----------------------------------
    # Channel 2's negative weights (reply -0.5, report -2.0) are exactly the
    # repulsive-influence assumption Mas & Flache (2013) and Takacs et al.
    # (2016) show has mixed, hard-to-identify support. Splitting the table
    # and switchable-zeroing the negative half makes that assumption
    # ablatable instead of load-bearing-and-invisible. `repulsion=False`
    # zeroes the negative weights outright — it does NOT clamp deltas, which
    # would introduce a rectification nonlinearity (a different model).
    social_weights_positive: tuple[tuple[str, float], ...] = (
        ("like", 1.0), ("repost", 1.5), ("quote", 0.5),
    )
    social_weights_negative: tuple[tuple[str, float], ...] = (
        ("reply", -0.5), ("report", -2.0),
    )
    repulsion: bool = True

    # -- C7: expression gate (spiral of silence) ---------------------------------
    # Hampton et al. (Pew 2014): perceived network disagreement predicts
    # self-censorship; Matthes et al. (2018) confirm a small-but-robust
    # opinion-support -> expression effect. 0 disables. Users respond to
    # *perceived* climate (their feed's composition, via perception.py's
    # blend), not the global sigma(t) — gating on the global state when a
    # local perception module exists would be the modelling error the
    # codebase is already structured to avoid.
    silence_gate: float = 0.0
    silence_conviction_moderation: float = 1.0

    # -- C8: reply contagion model -------------------------------------------------
    # The Hawkes process is a self-exciting SIMPLE contagion: intensity
    # depends on accumulated events, not distinct sources. Centola & Macy
    # (2007) and Centola (2010) show behaviours often need reinforcing
    # exposures from DISTINCT neighbours, and that long ties slow complex
    # contagion — inverting Granovetter. "hawkes" is current behaviour;
    # "threshold" makes reply propensity rise in the count of distinct
    # already-engaged in-neighbours.
    reply_model: str = "hawkes"               # hawkes | threshold
    # The threshold response curve: propensity ~ reply_prop * count^2 /
    # threshold_scale, clipped. Needs its own calibration to the §5.1
    # stylized facts before any D5 crossover result is reported — a config
    # field so that calibration is a sweep, not a code edit.
    threshold_scale: float = 16.0

    # -- C13: threads as digital micro publics --------------------------------
    # WHO replies. "kernel" sources reply candidates from the engagement
    # kernel's reply actions (users who saw the post and chose to reply —
    # content-sensitive), with the reply_prop lottery as a counted fallback
    # for posts that drew replies but no kernel candidate. "lottery" is the
    # old behaviour: authors drawn from the whole population, blind to
    # content. The Hawkes draw keeps owning WHEN and HOW MANY either way —
    # kernel replies firing on exposure would collapse reply timing onto the
    # exposure pass and lose the burstiness the Hawkes model exists to
    # produce. Applies to the hawkes path; the threshold reply model's
    # contagion candidate selection IS its mechanism and is not overridden.
    reply_selection: str = "kernel"           # kernel | lottery
    # C13b: reply intensity conditioned on the post's own dimensions (the
    # DMP claim: friction generates the discussion, not noise). mu_p =
    # mu_base * exp(gamma_prov*provocativeness + gamma_arousal*arousal +
    # gamma_disagree*||stance_p - stance_root||) — distance to the ROOT:
    # what makes a reply generative is that it contests what the arena
    # formed around. Empty tuple = flat mu (current behaviour).
    reply_mu_gamma: tuple[tuple[str, float], ...] = ()   # prov | arousal | disagree
    # C13c: which room low-conviction repliers conform to. "global" blends
    # toward sigma(t) (the platform-wide dominant stance, same weighting as
    # the root-post conformity line); "local" blends toward the thread's own
    # engagement-weighted mean stance — the argument they are actually in;
    # "blend" mixes the two by reply_conformity_mix. The local/global
    # contrast is itself a testable claim about where conformity pressure
    # comes from.
    reply_conformity: str = "global"          # local | global | blend
    reply_conformity_mix: float = 0.5         # only read when "blend"

    snapshot_every: int = 1
    exposure_sample_rate: float = 0.01

    # -- Experiment 03 package prerequisite: time-varying parameters ----------
    # "A schedule inside one config" (Experiment 03 design §5.1), so that
    # forked arms share a bit-identical prefix instead of being separate
    # configs that "merely start the same way" and diverge for reasons the
    # config hash cannot record. `((start_tick, ((field, value), ...)), ...)`,
    # sorted ascending on start_tick: at tick t, the LATEST entry with
    # start_tick <= t replaces the named fields on the base config wholesale
    # (not cumulatively on the previous entry's values — a withdrawal entry
    # states what it wants directly rather than undoing what came before).
    # Resolved by `effective_dynamics()` and read once per tick
    # (TickEngine.step), so every OTHER phase of that tick (including RNG
    # draws, which are keyed on `seed` alone, never on config content) is
    # unaffected by which arm is running — the prefix before the first
    # start_tick is bit-identical across arms that share it.
    #
    # Only fields TickEngine.step reads off its per-tick local (or rebuilds
    # from it, like the eight valence_* coefficients — see EndogenousValence
    # Params in tick.py's `step`) are schedule-reactive: inject_k, kernel/
    # kernel_theta/theta_scale, civility_prob, ranker, selection,
    # valence_mode and its coefficients, and similar. Fields consumed ONLY
    # in TickEngine.__post_init__ (kernel_learning, agreement_metric's
    # calibration, quality_trait_coupling) or read from the stored `Config`
    # directly rather than the per-tick local (apply_drift's hostility/
    # support/OU tables) are NOT schedule-reactive — scheduling them changes
    # nothing past construction. When in doubt, check where the field is
    # actually read; test_runner.py's bit-identical-prefix test is the
    # pattern for verifying a specific field either way.
    schedule: tuple[tuple[int, tuple[tuple[str, Any], ...]], ...] = ()

    def __post_init__(self) -> None:
        starts = [start for start, _ in self.schedule]
        if starts != sorted(starts) or len(starts) != len(set(starts)):
            raise ValueError(
                f"dynamics.schedule start ticks must be strictly increasing, got {starts!r}"
            )
        if any(start < 0 for start in starts):
            raise ValueError(f"dynamics.schedule start ticks must be >= 0, got {starts!r}")
        field_names = {f.name for f in dataclasses.fields(self)}
        for start, overrides in self.schedule:
            unknown = {name for name, _ in overrides} - field_names
            if unknown:
                raise ValueError(
                    f"dynamics.schedule entry at t={start} overrides unknown field(s) "
                    f"{sorted(unknown)!r}"
                )


def effective_dynamics(dynamics: "DynamicsConfig", t: int) -> "DynamicsConfig":
    """The `DynamicsConfig` in force at tick `t` under `dynamics.schedule`.

    Returns `dynamics` itself (not a copy) when the schedule is empty or
    nothing has started yet, so an unscheduled run pays no cost and no
    identity change. See `DynamicsConfig.schedule` for the contract.
    """
    active: tuple[tuple[str, Any], ...] | None = None
    for start, overrides in dynamics.schedule:
        if start <= t:
            active = overrides
        else:
            break  # __post_init__ guarantees ascending order
    if not active:
        return dynamics
    return dataclasses.replace(dynamics, **dict(active))


@dataclass(frozen=True)
class ScenarioConfig(Hashable):
    """Scenario layer, compatible with the stance editor's emitted JSON.

    Each axis: {name, pole_neg, pole_pos, marginal: {kind: empirical, bins,
    support, density}, expression_cost: {neg, pos}}.

    `name`, `pole_neg` and `pole_pos` are what make a result readable —
    "provision: leans market" rather than "stance_0 = -1.2" — and are consumed
    by `semantics.Lexicon`. `expression_cost` is written by the stance editor
    and read by nothing: it is reserved for an asymmetric expression-cost
    extension (it costs more to voice an unpopular position), not a bug.

    Note that a scenario *overrides* `population.stance_dims` through
    `Config.stance_dims()`. `data.scenario_config()` refuses the substitution
    unless asked, because silently changing D changes which mechanisms are
    even measurable.
    """

    name: str = "default"
    stance_axes: tuple[dict, ...] = ()
    topic_names: tuple[str, ...] = ()

    def __post_init__(self):
        for ax in self.stance_axes:
            d = ax.get("marginal", {}).get("density")
            if d is None:
                raise ValueError(f"axis {ax.get('name')!r} lacks an empirical marginal")
        if self.topic_names and len(self.topic_names) != len(set(self.topic_names)):
            raise ValueError("topic names must be unique")
        for name in self.topic_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"topic names must be non-empty strings, got {name!r}")

    @classmethod
    def from_editor_json(cls, data: dict, name: str = "scenario") -> "ScenarioConfig":
        scenario = data["scenario"]
        # topic_names was dropped here silently, which is why cfg.scenario
        # .topic_names has been declared, validated and empty since it was
        # added — nothing could ever populate it.
        return cls(
            name=name,
            stance_axes=tuple(scenario["stance_axes"]),
            topic_names=tuple(scenario.get("topic_names", ())),
        )

    def axis_count(self) -> int:
        return len(self.stance_axes)

    def axis_names(self) -> tuple[str, ...]:
        return tuple(str(ax.get("name", i)) for i, ax in enumerate(self.stance_axes))

    def poles(self) -> tuple[tuple[str, str], ...]:
        """`(negative, positive)` pole label per axis, in axis order."""
        return tuple(
            (str(ax.get("pole_neg", "-")), str(ax.get("pole_pos", "+")))
            for ax in self.stance_axes
        )


@dataclass(frozen=True)
class WorldConfig(Hashable):
    """LLM realization (spec §2.10, dev §6 step 12) — offline only, never
    inside the tick. Model choice defaults to an Ollama Cloud model
    (https://ollama.com); `OllamaCloudClient` reads the API key from
    `OLLAMA_API_KEY`, not from here.
    """

    llm_model: str = "gpt-oss:120b-cloud"
    temperature: float = 0.8
    voice_card_max_tokens: int = 220
    render_max_tokens: int = 120
    render_batch_size: int = 30          # posts per rendering call (spec: 20-50)
    n_bands: int = 5                     # trait quantization band count (spec §2.10)

    # channel 3 (LLM adjudication) event gating — rare, event-triggered
    adjudication_top_percentile: float = 0.99   # top 1% engagement
    adjudication_pile_on_threshold: int = 20    # hostile replies received
    adjudication_max_delta: float = 0.1         # clip(Δ_llm, -eps, eps)


@dataclass(frozen=True)
class Config(Hashable):
    # Bumped with every change that forks the run cache. All C1-C10 config
    # surface landed in one commit (change spec §0.1) so the cache forks
    # once, legibly, rather than ten times; C13's reply-path surface is the
    # second batched fork.
    schema_version: int = 3
    population: PopulationConfig = field(default_factory=PopulationConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    label: str = "default"

    def __post_init__(self):
        n_topics = self.population.n_topics
        if len(self.scenario.topic_names) > n_topics:
            raise ValueError(
                f"scenario names {len(self.scenario.topic_names)} topics but "
                f"population.n_topics is {n_topics}"
            )

    def stance_dims(self) -> int:
        n = self.scenario.axis_count()
        if n > 0:
            return n
        return max(1, self.population.stance_dims)

    def sub_hashes(self) -> dict[str, str]:
        return {
            "population": self.population.hash(),
            "graph": self.graph.hash(),
            "dynamics": self.dynamics.hash(),
            "scenario": self.scenario.hash(),
            "world": self.world.hash(),
        }
