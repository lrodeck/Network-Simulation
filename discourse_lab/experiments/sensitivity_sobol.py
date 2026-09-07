"""C10.2 — Sobol sensitivity over the unanchored parameters (change spec).

Several parameters landed with C1-C8 carrying no empirical anchor at all:
`lr_affect`, `affect_ou_k`, `selection_beta`, `silence_gate`, `lr_kernel`,
`kernel_gain_ou_k`, `drift_lr_behavior`. Any result driven by a
high-sensitivity, unanchored parameter should be reported with that fact
attached; this module produces the first-order (S1) and total-order (ST)
indices that make the attachment honest.

Variance-based Sobol decomposition over a Saltelli sample. Each of the
N * (k + 2) model points is a full run per seed, so the cost is
`n_samples * (k + 2) * n_seeds` runs — start small (n=8, k=7, 2 seeds is
~112 runs) and only scale up the parameters that survive.

Parameters are scaled multiplicatively around their base value (the same
±50% plausibility band `analysis.sensitivity` uses), so the indices are
dimensionless and comparable across parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from discourse_lab.analysis import get_param, set_param

# C1-C8 parameters with no empirical anchor (change spec C10.2's list, plus
# the affect/kernel-learning rates that joined them).
UNANCHORED_PARAMS = (
    "dynamics.lr_affect",
    "dynamics.affect_ou_k",
    "dynamics.drift_lr_behavior",
    "dynamics.lr_kernel",
    "dynamics.kernel_gain_ou_k",
    "dynamics.silence_gate",
)


@dataclass
class SobolReport:
    indices: pl.DataFrame          # param, S1, ST
    metric: str
    n_samples: int
    n_seeds: int
    var_total: float


def _saltelli_matrices(k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform draws on (0,1) in Saltelli's A/B/C arrangement: N rows for A,
    N for B, then N cross rows per parameter (A with column i from B)."""
    A = rng.random((n, k))
    B = rng.random((n, k))
    blocks = [A, B] + [A.copy() for _ in range(k)]
    for i in range(k):
        blocks[2 + i][:, i] = B[:, i]
    return np.concatenate(blocks, axis=0)  # (N*(k+2), k)


def _to_value(base_value: float, u: float, band: float = 0.5) -> float:
    """Map the unit interval onto base * (1 ± band), keeping the value
    positive — the parameters above are rates and gates, where 0 and
    negatives are outside the model's intended regime."""
    lo, hi = base_value * (1 - band), base_value * (1 + band)
    return float(lo + u * (hi - lo))


def sobol_indices(
    base,
    seeds,
    params=UNANCHORED_PARAMS,
    n_samples: int = 8,
    metric: str = "n_engagements",
    band: float = 0.5,
    seed: int = 0,
) -> SobolReport:
    """S1/ST for each parameter against `metric` (mean over ticks per run)."""
    rng = np.random.default_rng(seed)
    k = len(params)
    units = _saltelli_matrices(k, n_samples, rng)
    base_values = [float(get_param(base, p)) for p in params]

    rows = []
    for row in units:
        cfg = base
        for p, bv, u in zip(params, base_values, row):
            cfg = set_param(cfg, p, _to_value(bv, u, band))
        vals = []
        for s in seeds:
            from discourse_lab.runner import cached_run, load_run

            cached_run(cfg, s)
            frame = load_run(cfg, s).metrics()
            if metric not in frame.columns:
                raise KeyError(f"metric {metric!r} not in metrics.parquet")
            v = frame[metric].to_numpy().astype(float)
            v = v[np.isfinite(v)]
            vals.append(float(v.mean()) if len(v) else float("nan"))
        rows.append(float(np.nanmean(vals)))

    Y = np.asarray(rows)
    n = n_samples
    y_a, y_b = Y[:n], Y[n : 2 * n]
    y_c = [Y[2 * n + i * n : 2 * n + (i + 1) * n] for i in range(k)]
    var_total = float(np.var(np.concatenate([y_a, y_b]), ddof=1))
    if var_total < 1e-12:
        out = pl.DataFrame({"param": list(params), "S1": [0.0] * k, "ST": [0.0] * k})
        return SobolReport(out, metric, n, len(seeds), var_total)

    indices = []
    for i in range(k):
        s1 = float(np.mean(y_b * (y_c[i] - y_a)) / var_total)
        st = float(0.5 * np.mean((y_b - y_c[i]) ** 2) / var_total)
        indices.append({"param": params[i], "S1": s1, "ST": st})
    return SobolReport(pl.DataFrame(indices), metric, n, len(seeds), var_total)
