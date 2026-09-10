"""Engagement valence (change spec V1): two independent axes assigned at the
moment of engagement.

    agree / disagree    position -- derived from the kernel's own
                        `agreement` feature (exposure/kernel.py), so a
                        supportive cross-camp reply and a cross-camp
                        quote-dunk are no longer the same event
    civil / hostile     conduct -- exogenous for V1/V2 (a coin flip at
                        `civility_prob`); V3 replaces it with a per-user
                        logit on animus and stance distance

Orthogonal by construction (spec: "Agreement is about position; civility is
about conduct. They are orthogonal and must not be collapsed"): pile-on
(agree, in structure, on the in-group side, but hostile in conduct toward the
out-group) and a civil cross-camp disagreement are both representable, which
the pre-V1 action-keyed hostility table could not do — a reply was a reply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The four cells, named consistently with `DynamicsConfig.affect_valence_signs`.
VALENCE_CELLS = ("disagree_hostile", "disagree_civil", "agree_civil", "agree_hostile")


@dataclass(frozen=True)
class EngagementValence:
    """One row per exposure. Both fields are boolean arrays of equal length."""

    agree: np.ndarray
    civil: np.ndarray

    def __len__(self) -> int:
        return len(self.agree)

    def cell(self) -> np.ndarray:
        """The named cell per row, for persistence/inspection."""
        return np.where(
            self.agree,
            np.where(self.civil, "agree_civil", "agree_hostile"),
            np.where(self.civil, "disagree_civil", "disagree_hostile"),
        )


def assign_valence(
    agreement: np.ndarray,
    agree_delta: float,
    rng: np.random.Generator,
    civility_prob: float = 0.5,
    force_agree: bool | None = None,
    force_civil: bool | None = None,
) -> EngagementValence:
    """`agreement` is the kernel's own `-distance` feature
    (`exposure/kernel.py::compute_features`), one row per exposure — always
    <= 0, more negative meaning farther apart. `agree_delta` is an ABSOLUTE
    distance threshold, calibrated once per population (`TickEngine.
    __post_init__`, the same median-pairwise-distance technique
    `outcomes.py::selection_filtering` already uses for its own `delta`) —
    NOT a per-batch median split. A per-batch split was tried first and
    rejected: it forces exactly half of every exposure batch to read as
    "agree" no matter how one-sided the actual mix is, which washed out
    exactly the asymmetry a kernel like `outrage` exists to create (it
    deliberately targets disagreeable out-group content — most of a batch
    SHOULD read as disagree, not a coin-flip half).

    `force_agree` / `force_civil` pin that axis to a constant for every row
    — the fixture hook the V1/V2 conformance tests need to hold "every
    engagement is agree+civil" fixed while sweeping the other run, without
    threading a probability through a whole population.
    """
    m = len(agreement)
    if force_agree is not None:
        agree = np.full(m, bool(force_agree))
    else:
        agree = agreement >= -agree_delta

    if force_civil is not None:
        civil = np.full(m, bool(force_civil))
    else:
        civil = rng.random(m) < civility_prob

    return EngagementValence(agree=agree, civil=civil)


@dataclass(frozen=True)
class EndogenousValenceParams:
    """V3's ~8 free coefficients (of the spec's 15-20 total, the rest being
    V3's own kernel-side terms elsewhere). None are empirically anchored —
    same status as `affect_ou_k` — but the SIGNS below are fixed by theory,
    matching the project's convention for the hostility table itself:

        beta_dist    -  agreement falls with dyad distance (the obvious sign)
        beta_ident   -  "openness is identification, not recent drift" (spec):
                        stronger attachment to one's own camp makes a user
                        LESS likely to register cross-camp content as
                        agreement
        gamma_animus -  higher animus makes a user LESS civil -- the
                        self-reinforcement term the bistability probe needs;
                        magnitude is what the sweep in
                        `experiments/sensitivity_sobol.py` must cover,
                        including the region where it is too weak to
                        produce two basins (the non-tautology requirement)
        beta0/gamma0/beta_dist... others default to 0 (no baseline skew,
        no distance effect on civility) -- genuinely free, swept params.
    """

    beta0: float = 0.0
    beta_dist: float = -1.0
    beta_ident: float = -0.3
    noise_agree: float = 1.0
    gamma0: float = 0.0
    gamma_animus: float = -1.0
    gamma_dist: float = 0.0
    noise_civil: float = 1.0


def assign_valence_endogenous(
    agreement: np.ndarray,
    animus_i: np.ndarray,
    identification_i: np.ndarray,
    rng: np.random.Generator,
    params: EndogenousValenceParams,
    force_agree: bool | None = None,
    force_civil: bool | None = None,
) -> EngagementValence:
    """V3: valence produced by the engaging user's OWN state and the dyad's
    geometry, not an exogenous coin flip -- the feedback loop the bistability
    question needs (does a population already in a hostile regime metabolize
    added contact as attack).

        P(agree) = sigma( beta0 + beta_dist * d(s_i, s_j) + beta_ident * identification_i + eps )
        P(civil) = sigma( gamma0 + gamma_animus * animus_i          + gamma_dist * d(s_i, s_j) + eta )

    `d(s_i, s_j)` is recovered from the kernel's own `agreement` feature
    (`= -d`), so this stays in the same units V1/V2 already use and over the
    FULL per-axis stance vector (V6(1)'s requirement) rather than a
    camp-membership flag -- a user close on one axis and far on another is
    representable exactly as the "Republican liking Sanders" case requires.
    `animus_i` / `identification_i` are the engaging user's own state, used
    the same way `exposure/kernel.py`'s `outgroup_x_animus` /
    `ingroup_x_ident` features already do (no renormalization). `eps`/`eta`
    are real noise, added inside the logit per the spec's own formula: the
    predictors set the skew, the draw against the resulting probability is
    what keeps the outcome genuinely stochastic rather than a hard
    threshold on observables.

    `force_agree` / `force_civil` pin an axis for every row, same fixture
    hook `assign_valence` (V1/V2's exogenous path) provides.
    """
    from scipy.special import expit

    m = len(agreement)
    dist = -agreement

    if force_agree is not None:
        agree = np.full(m, bool(force_agree))
    else:
        eps = rng.normal(0.0, params.noise_agree, m)
        agree_logit = params.beta0 + params.beta_dist * dist + params.beta_ident * identification_i + eps
        agree = rng.random(m) < expit(agree_logit)

    if force_civil is not None:
        civil = np.full(m, bool(force_civil))
    else:
        eta = rng.normal(0.0, params.noise_civil, m)
        civil_logit = params.gamma0 + params.gamma_animus * animus_i + params.gamma_dist * dist + eta
        civil = rng.random(m) < expit(civil_logit)

    return EngagementValence(agree=agree, civil=civil)


def valence_cell_signs(
    agree: np.ndarray, civil: np.ndarray, signs: dict[str, float]
) -> np.ndarray:
    """`affect_valence_signs` applied elementwise: `weight(action) *
    valence_sign_and_magnitude(cell)` is assembled by the caller (drift.py)
    — this returns just the cell factor.
    """
    disagree_hostile = signs.get("disagree_hostile", 0.0)
    disagree_civil = signs.get("disagree_civil", 0.0)
    agree_civil = signs.get("agree_civil", 0.0)
    agree_hostile = signs.get("agree_hostile", 0.0)
    return np.where(
        agree,
        np.where(civil, agree_civil, agree_hostile),
        np.where(civil, disagree_civil, disagree_hostile),
    )
