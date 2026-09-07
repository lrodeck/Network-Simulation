"""The selection layer (change spec C2.1): a content-conditional attention
stage applied after ranking and before the engagement kernel.

    P(attend | exposed) = sigma( beta_pos * pos_decay + beta_agree * agreement
                                 + beta_arousal * arousal )

Bakshy, Messing & Adamic (2015) found individual choice filtered
cross-cutting content *more than the algorithm did* (~17%/6% self-selection
vs 5%/8% algorithmic, by group). The model previously had no stage between
exposure and reaction, so the strongest empirically-supported echo-chamber
mechanism was structurally absent — and every ranker result implicitly
attributed to the algorithm what the field attributes mostly to users.

Implementations, each a claim:

    position_only    current behaviour: attention was already rationed by the
                     position decay in exposure/attention.py, so selection is
                     a pass-through. The default; preserves existing runs.
    homophilous      users click toward agreement (Bakshy).
    arousal_seeking  users click toward arousal regardless of agreement.

Runs in its own `selection` phase with its own RNG stream: it draws, and
folding it into `exposure` would shift the exposure draws and confound every
sweep (change spec §0.2).

`position_decay` is passed in even though the budget/decay stage already
thinned by it — the logit wants it as a *term* (a click is more likely near
the top of the feed even conditional on exposure), which is a different
statement than the thinning and composes with it.
"""

from __future__ import annotations

import numpy as np

from discourse_lab.exposure.attention import Exposures
from discourse_lab.registry import get, names, register
from scipy.special import expit

# (name, default) betas for the selection logit; `selection_beta` in config
# overrides by name.
DEFAULT_BETAS: dict[str, float] = {
    "beta_pos": 2.0,
    "beta_agree": 1.0,
    "beta_arousal": 0.5,
}


def _betas(overrides) -> dict[str, float]:
    betas = dict(DEFAULT_BETAS)
    for name, value in overrides or ():
        if name not in betas:
            raise KeyError(
                f"unknown selection beta {name!r}; expected one of {sorted(betas)}"
            )
        betas[name] = float(value)
    return betas


def selection_names() -> list[str]:
    return names("selection")


def apply_selection(
    name: str,
    exposures: Exposures,
    features: dict[str, np.ndarray],
    tau_position: float,
    rng: np.random.Generator,
    beta_overrides=(),
) -> np.ndarray:
    """Boolean attend-mask over the exposure rows, one draw per exposure.

    The mask is applied AFTER ranking and BEFORE the kernel, so the kernel
    only ever sees attended pairs; the exposure log records both `exposed`
    and `attended` per sampled row, which is what keeps the echo-chamber
    index unambiguous about which quantity it measures.
    """
    m = len(exposures)
    if name == "position_only":
        return np.ones(m, dtype=bool)

    betas = _betas(beta_overrides)
    pos_decay = np.exp(-exposures.rank / tau_position)
    logit = betas["beta_pos"] * pos_decay
    if name == "homophilous":
        logit = logit + betas["beta_agree"] * features["agreement"]
    elif name == "arousal_seeking":
        logit = logit + betas["beta_arousal"] * features["arousal"]
    else:
        raise KeyError(
            f"unknown selection model {name!r}; expected one of {selection_names()}"
        )
    # position_only is the degenerate case of the same logit with the
    # content betas at zero, so the alternatives are strictly additive
    # amendments to it rather than a different functional form.
    return rng.random(m) < expit(logit)


@register("selection", "position_only")
def _position_only(exposures, features, tau_position, rng, beta_overrides=()):
    return apply_selection("position_only", exposures, features, tau_position, rng, beta_overrides)


@register("selection", "homophilous")
def _homophilous(exposures, features, tau_position, rng, beta_overrides=()):
    return apply_selection("homophilous", exposures, features, tau_position, rng, beta_overrides)


@register("selection", "arousal_seeking")
def _arousal_seeking(exposures, features, tau_position, rng, beta_overrides=()):
    return apply_selection("arousal_seeking", exposures, features, tau_position, rng, beta_overrides)


def named_selection(name: str):
    return get("selection", name)
