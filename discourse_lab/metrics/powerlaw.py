"""Discrete power-law fit (spec §5.1: engagement per post, heavy-tailed with
alpha in [2, 3]).

Engagement counts are integers, so this is the *discrete* Clauset–Shalizi–
Newman estimator, not the continuous one. The continuous MLE applied to small
integer counts is biased badly upward — at the counts this simulation
produces (medians near zero, most posts in single digits) that bias is the
whole answer, so it matters.

    ln L = -alpha * sum(ln x_i) - n * ln(zeta(alpha, x_min))

is maximised numerically for alpha. x_min is chosen by minimising the KS
distance between the empirical CDF and the discrete power-law CDF, which uses
the Hurwitz zeta function:

    P(X >= x) = zeta(alpha, x) / zeta(alpha, x_min)

No `powerlaw` package — scipy.special.zeta is all that is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta

# Below this many points in the tail the estimate is noise. CSN recommend
# n >= 50 before quoting alpha at all; we return nan rather than a number
# that looks authoritative and is not.
MIN_TAIL = 50


@dataclass(frozen=True)
class PowerLawFit:
    alpha: float
    xmin: float
    ks: float          # KS distance at the chosen xmin
    n_tail: int        # points at or above xmin

    def __bool__(self) -> bool:
        return bool(np.isfinite(self.alpha))


def _alpha_mle(tail: np.ndarray, xmin: float) -> float:
    """Exact discrete MLE: maximise the discrete log-likelihood

        ln L = -alpha * sum(ln x_i) - n * ln(zeta(alpha, x_min))

    numerically. The closed-form continuous approximation (CSN eq. 3.7, with
    the half-integer correction) is cheaper but measurably biased low on the
    small integer counts this simulation produces — around 5-10% at the
    tail sizes seen in practice, which is the difference between landing
    inside the spec's alpha range and outside it. The optimisation is
    one-dimensional and bounded, so the cost is irrelevant.
    """
    n = len(tail)
    sum_log = float(np.sum(np.log(tail)))

    def negative_log_likelihood(alpha: float) -> float:
        norm = zeta(alpha, xmin)
        if not np.isfinite(norm) or norm <= 0:
            return np.inf
        return alpha * sum_log + n * np.log(norm)

    result = minimize_scalar(negative_log_likelihood, bounds=(1.01, 10.0), method="bounded")
    return float(result.x) if result.success else float("nan")


def _ks_distance(tail: np.ndarray, xmin: float, alpha: float) -> float:
    """KS distance between the empirical CCDF and the discrete power law."""
    x = np.sort(tail)
    norm = zeta(alpha, xmin)
    if not np.isfinite(norm) or norm <= 0:
        return np.inf
    theoretical_cdf = 1.0 - zeta(alpha, x) / norm
    empirical_cdf = np.arange(1, len(x) + 1) / len(x)
    return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))


def powerlaw_fit(x: np.ndarray, xmin: float | None = None) -> PowerLawFit:
    """Fit the tail of `x` and return alpha with the x_min it was fitted above.

    `x_min` is swept over the distinct observed values when not supplied,
    picking the one that minimises KS distance. Values below 1 are dropped —
    a discrete power law is not defined at or below zero, and posts with zero
    engagement are the body of the distribution, not the tail.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 1)]
    if len(x) < MIN_TAIL:
        return PowerLawFit(alpha=float("nan"), xmin=float("nan"), ks=float("nan"), n_tail=len(x))

    if xmin is not None:
        tail = x[x >= xmin]
        if len(tail) < MIN_TAIL:
            return PowerLawFit(float("nan"), float(xmin), float("nan"), len(tail))
        alpha = _alpha_mle(tail, xmin)
        return PowerLawFit(float(alpha), float(xmin), _ks_distance(tail, xmin, alpha), len(tail))

    # sweep candidate x_min over distinct observed values, leaving enough tail
    candidates = np.unique(x)
    candidates = candidates[candidates >= 1]

    best = PowerLawFit(float("nan"), float("nan"), float("inf"), 0)
    for candidate in candidates:
        tail = x[x >= candidate]
        if len(tail) < MIN_TAIL:
            break  # candidates are ascending, so every later one is smaller still
        alpha = _alpha_mle(tail, candidate)
        if not np.isfinite(alpha) or alpha <= 1:
            continue
        ks = _ks_distance(tail, candidate, alpha)
        if ks < best.ks:
            best = PowerLawFit(float(alpha), float(candidate), ks, len(tail))

    return best


def powerlaw_alpha(x: np.ndarray, xmin: float | None = None) -> float:
    """Just the exponent, for callers that do not need the diagnostics."""
    return powerlaw_fit(x, xmin=xmin).alpha


def ccdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical complementary CDF, `(values, P(X >= value))` — the standard
    way to eyeball a heavy tail on log-log axes, and what the distribution
    figures plot.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([]), np.array([])
    values = np.sort(x)
    survival = 1.0 - np.arange(len(values)) / len(values)
    return values, survival
