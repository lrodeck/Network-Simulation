"""Stored-space links (spec §1.1): drift operates on `x_stored`, an
unconstrained real; `x_used = link(x_stored)` enforces the constraint on read.

    x_used   in [0,1]  -> stored as a logit
    x_used   in (0,∞)  -> stored as a log
    x_used   unbounded -> stored as itself

Marginals are drawn in used-space (that is what `F_i^{-1}` targets); the
inverse link then gives the stored value that reproduces that draw.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit as _logit

LINKS: dict[str, tuple] = {
    # name: (stored -> used, used -> stored)
    "identity": (lambda s: s, lambda u: u),
    "logit": (lambda s: expit(s), lambda u: _logit(np.clip(u, 1e-9, 1 - 1e-9))),
    "log": (lambda s: np.exp(s), lambda u: np.log(np.clip(u, 1e-12, None))),
    # symmetric bounded traits (e.g. post valence in [-1,1]) use tanh, the
    # natural analogue of the logit for a [0,1] trait.
    "tanh": (lambda s: np.tanh(s), lambda u: np.arctanh(np.clip(u, -1 + 1e-9, 1 - 1e-9))),
}


def to_used(stored: np.ndarray, link: str) -> np.ndarray:
    return LINKS[link][0](stored)


def to_stored(used: np.ndarray, link: str) -> np.ndarray:
    return LINKS[link][1](used)
