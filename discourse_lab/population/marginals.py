"""Marginal registry (spec §2.1) and the empirical inverse CDF.

Every marginal exposes `icdf(w) -> x` for `w` uniform on `(0,1)`, so the
copula pipeline (`w = Φ(z)`, `x = F^-1(w)`) is a single dispatch regardless of
which family produced the target distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

from discourse_lab.registry import get, names, register


@dataclass(frozen=True)
class Marginal:
    icdf: Callable[[np.ndarray], np.ndarray]


def _clip01(w: np.ndarray) -> np.ndarray:
    return np.clip(w, 1e-12, 1 - 1e-12)


@register("marginal", "normal")
def normal(mu: float = 0.0, sigma: float = 1.0) -> Marginal:
    dist = stats.norm(loc=mu, scale=sigma)
    return Marginal(icdf=lambda w: dist.ppf(_clip01(w)))


@register("marginal", "lognormal")
def lognormal(mu: float = 0.0, sigma: float = 1.2) -> Marginal:
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    return Marginal(icdf=lambda w: dist.ppf(_clip01(w)))


@register("marginal", "pareto")
def pareto(alpha: float = 2.3, scale: float = 1.0) -> Marginal:
    dist = stats.pareto(b=alpha, scale=scale)
    return Marginal(icdf=lambda w: dist.ppf(_clip01(w)))


@register("marginal", "beta")
def beta(a: float = 2.0, b: float = 2.0) -> Marginal:
    dist = stats.beta(a, b)
    return Marginal(icdf=lambda w: dist.ppf(_clip01(w)))


@register("marginal", "vonmises")
def vonmises(mu: float = 0.0, kappa: float = 2.0) -> Marginal:
    dist = stats.vonmises(kappa=kappa, loc=mu)
    return Marginal(icdf=lambda w: dist.ppf(_clip01(w)))


def empirical_from_editor(bins: int, support: tuple[float, float], density: list[float]) -> Marginal:
    """Piecewise-linear inverse CDF from the stance editor's density array
    (dev §6 step 3): `bins` samples of a PDF over a uniform grid on `support`.
    """
    lo, hi = support
    density = np.asarray(density, dtype=float)
    if density.shape[0] != bins:
        raise ValueError(f"density has {density.shape[0]} entries, expected {bins}")
    if np.any(density < 0):
        raise ValueError("density must be non-negative")

    edges = np.linspace(lo, hi, bins + 1)
    bin_width = (hi - lo) / bins
    cdf_edges = np.concatenate(([0.0], np.cumsum(density) * bin_width))
    total = cdf_edges[-1]
    if total <= 0:
        raise ValueError("density integrates to zero")
    cdf_edges = cdf_edges / total

    def icdf(w: np.ndarray) -> np.ndarray:
        return np.interp(_clip01(w), cdf_edges, edges)

    return Marginal(icdf=icdf)


def marginal_names() -> list[str]:
    return names("marginal")


def build_marginal(name: str, **params) -> Marginal:
    return get("marginal", name)(**params)
