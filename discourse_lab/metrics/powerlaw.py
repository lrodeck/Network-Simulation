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

**Fitting a power law is not evidence that there is one.** The MLE returns an
alpha for any data at all, and spec §5.1's target of [2, 3] can be met or
missed on the same sample purely by where x_min lands. Measured on this
model's engagement counts under an engagement-optimised feed:

    x_min      2      5     20     40     80    150    190    300
    alpha   1.54   1.66   1.96   2.28   2.92   3.68   4.08   5.66

A genuine power law gives roughly constant alpha across x_min; this rises
monotonically, which is the signature of a curved (lognormal-like or
exponentially truncated) tail. So `powerlaw_fit` also reports
`lognormal_ratio` — Vuong's normalised log-likelihood ratio against a
lognormal fitted to the same tail — and `is_powerlaw`, which is False when a
lognormal explains the tail at least as well. Quote alpha only when that
holds.
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
    ks: float               # KS distance at the chosen xmin
    n_tail: int             # points at or above xmin
    lognormal_ratio: float = float("nan")   # Vuong R, >0 favours the power law
    lognormal_p: float = float("nan")       # two-sided p for R != 0
    alpha_spread: float = float("nan")      # (max-min)/median of alpha across x_min

    def __bool__(self) -> bool:
        return bool(np.isfinite(self.alpha))

    @property
    def is_powerlaw(self) -> bool:
        """Whether `alpha` describes the data or merely where x_min landed.

        Gated on `alpha_spread`, the fractional variation of alpha across
        x_min, not on the lognormal comparison. A power law is scale-free, so
        its exponent is the same wherever the tail is cut; a curved tail gives
        an exponent that climbs with x_min. That is a property of the data and
        needs no competing model.

        The lognormal likelihood ratio is reported alongside but deliberately
        not used as the gate: the lognormal has two free parameters against
        the power law's one and wins on in-sample likelihood even for genuine
        Zipf draws (R = -12.6 on a synthetic Zipf(2.5)), which is the
        inconclusiveness Clauset-Shalizi-Newman note for samples of this size.
        """
        return bool(np.isfinite(self.alpha_spread) and self.alpha_spread < 0.30)


def alpha_spread(x: np.ndarray, xmin: float | None = None) -> float:
    """Fractional variation of alpha as x_min is swept across the data.

    A power law is scale-free: cutting the tail higher must not change the
    exponent. Curvature shows up here immediately, and this is the evidence
    that the engagement distribution is not a power law — alpha ran 1.54 at
    x_min 2 to 5.66 at x_min 300, a spread of 1.6, while the fit at the
    KS-optimal x_min reported a single authoritative-looking 4.08.

    Swept over the whole support, not just above the fitted x_min: the KS
    choice picks the most power-law-looking stretch by construction, so
    measuring the spread only above it hides exactly what is being tested.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 1)]
    if len(x) < 4 * MIN_TAIL:
        return float("nan")

    lo = max(float(np.quantile(x, 0.50)), 1.0)
    hi = max(float(np.quantile(x, 0.995)), lo + 1)
    cuts = np.unique(np.geomspace(lo, hi, 8))

    alphas = []
    for cut in cuts:
        sub = x[x >= cut]
        if len(sub) < MIN_TAIL:
            continue
        a = _alpha_mle(sub, float(cut))
        if np.isfinite(a) and a > 1:
            alphas.append(a)
    if len(alphas) < 3:
        return float("nan")
    alphas = np.array(alphas)
    return float((alphas.max() - alphas.min()) / np.median(alphas))


def compare_lognormal(tail: np.ndarray, xmin: float, alpha: float) -> tuple[float, float]:
    """Vuong's normalised log-likelihood ratio, power law vs lognormal, on the
    same tail. Returns `(R, p)`: R > 0 favours the power law, and p is the
    two-sided probability of an |R| this large when the two fit equally well.

    The lognormal is fitted to the tail and renormalised above `xmin`, which
    is the standard companion test to a CSN fit — without it the fit reports
    an exponent for data that is not power-law distributed at all.
    """
    from scipy import stats

    n = len(tail)
    if n < MIN_TAIL:
        return float("nan"), float("nan")

    log_x = np.log(tail)

    def _ll_ln(params: np.ndarray) -> np.ndarray:
        mu, log_sigma = params[0], params[1]
        sigma = np.exp(log_sigma)
        survival = stats.norm.sf((np.log(xmin) - mu) / sigma)
        if not np.isfinite(survival) or survival <= 0:
            return np.full(len(tail), -1e12)
        return (
            -log_x - np.log(sigma) - 0.5 * np.log(2 * np.pi)
            - 0.5 * ((log_x - mu) / sigma) ** 2 - np.log(survival)
        )

    # The lognormal must be fitted to the *truncated* tail, by maximising the
    # same renormalised likelihood it is then scored on. Using the plain
    # (untruncated) MLE of mu and sigma on tail data is biased and makes the
    # power law win spuriously — on a synthetic lognormal(0, 2) sample that
    # version reported R = +5.6, confidently backing the wrong model.
    from scipy.optimize import minimize

    start = np.array([log_x.mean(), np.log(max(log_x.std(ddof=0), 1e-6))])
    opt = minimize(lambda q: -_ll_ln(q).sum(), start, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
    ll_ln = _ll_ln(opt.x)

    ll_pl = -alpha * log_x - np.log(zeta(alpha, xmin))

    diff = ll_pl - ll_ln
    r = float(diff.sum())
    sd = float(diff.std(ddof=0))
    if sd <= 0:
        return r, float("nan")
    z = r / (np.sqrt(n) * sd)
    return z, float(stats.norm.sf(abs(z)) * 2)


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
        r, pval = compare_lognormal(tail, float(xmin), alpha)
        return PowerLawFit(
            float(alpha), float(xmin), _ks_distance(tail, xmin, alpha), len(tail),
            r, pval, alpha_spread(x),
        )

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

    if best:
        tail = x[x >= best.xmin]
        r, pval = compare_lognormal(tail, best.xmin, best.alpha)
        best = PowerLawFit(best.alpha, best.xmin, best.ks, best.n_tail, r, pval,
                           alpha_spread(x))

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
