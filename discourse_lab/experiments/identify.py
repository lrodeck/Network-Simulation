"""C10.1 — identifiability: are two theories distinguishable from the
metrics the model reports?

Equifinality is this model's central epistemic risk: many mechanisms produce
the same macro pattern, and matching stylized facts is necessary but not
sufficient for validation (Grimm et al.'s ABM-validation critique). C5
raises the concrete case: `outrage` and `homophily` may be non-separable on
any observable the model produces, because both predict engagement
concentrated within camp.

The report is deliberately two-part: AUC **and** which metric carries the
signal. "These theories are separable, by this observable" is a much
stronger claim than an AUC alone — and AUC ≈ 0.5 is itself a reportable
result (publishable non-identification), not a failure of the harness.

The classifier is a plain L2-regularised logistic regression fitted by
gradient descent: no new dependency, deterministic given the seed, and
honest about being linear — if a linear read of the reported metrics cannot
tell two theories apart, the burden is on the theorist to name the
nonlinear observable, not on the harness to find one by force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discourse_lab.analysis import MIN_SEEDS


@dataclass
class SeparabilityReport:
    auc: float
    top_metric: str
    importances: dict[str, float]
    n_a: int
    n_b: int


def _fit_logistic(X: np.ndarray, y: np.ndarray, seed: int, iters: int = 4000, lr: float = 0.1):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, d)
    b = 0.0
    lam = 1e-2
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (p - y) @ X / n + lam * w
        gb = (p - y).mean()
        w -= lr * grad
        b -= lr * gb
    return w, b


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), tie-corrected."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _standardize(X: np.ndarray, mean=None, sd=None):
    mean = X.mean(axis=0) if mean is None else mean
    sd = X.std(axis=0) if sd is None else sd
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mean) / sd, mean, sd


def separability(
    runs_a: np.ndarray,
    runs_b: np.ndarray,
    metrics: list[str],
    seed: int = 0,
    folds: int = 5,
) -> SeparabilityReport:
    """Cross-validated AUC of a logistic classifier separating two
    conditions, plus permutation importance per metric.

    `runs_a` / `runs_b` are (n_runs, n_metrics) arrays of per-run summary
    metrics — e.g. the mean over ticks of `metrics.parquet` columns, or
    `normative_outcomes` values, one row per seed per condition. The claim
    the caller gets to make:

        AUC ~ 1.0  -> separable; `top_metric` names the observable that
                      carries it
        AUC ~ 0.5  -> NOT separable on the reported metrics — report it as
                      such and treat "find a separating observable" as the
                      open problem. Do not tune features until the AUC is
                      high; that is fitting the model to produce a
                      distinction the data may not support.
    """
    runs_a = np.asarray(runs_a, dtype=float)
    runs_b = np.asarray(runs_b, dtype=float)
    if len(runs_a) < MIN_SEEDS or len(runs_b) < MIN_SEEDS:
        raise ValueError(
            f"separability needs at least {MIN_SEEDS} seeds per condition (spec §4.4) "
            f"to cross-validate over {folds} folds without a fold landing single-class; "
            f"got {len(runs_a)} and {len(runs_b)}. A returned AUC==nan here is not a "
            "non-identification result — it means the folds could not be evaluated."
        )
    X = np.vstack([runs_a, runs_b])
    y = np.concatenate([np.zeros(len(runs_a)), np.ones(len(runs_b))])
    X, mean, sd = _standardize(X)

    # cross-validated AUC over shuffled folds
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    aucs = []
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        w, b = _fit_logistic(X[train], y[train], seed + f)
        aucs.append(_auc(X[test] @ w + b, y[test]))
    auc = float(np.mean(aucs))

    # permutation importance: how much does test AUC drop when a metric's
    # column is shuffled (its information destroyed)?
    w, b = _fit_logistic(X, y, seed)
    base = _auc(X @ w + b, y)
    importances = {}
    for j, name in enumerate(metrics):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        importances[name] = float(base - _auc(Xp @ w + b, y))
    top = max(importances, key=importances.get) if importances else ""
    return SeparabilityReport(auc=auc, top_metric=top, importances=importances,
                              n_a=len(runs_a), n_b=len(runs_b))
