"""Post-run analysis (spec §5.1-5.3, dev §6 step 9): a separate module over a
completed `Run` — trajectories, distributions, cross-seed comparison. Unlike
`measures/` (live, per-tick, cheap), these read whatever data is available
after the fact and can be as expensive as the question demands.

Every reported effect should be a difference against a matched null
(§5.3): same population, same graph, same activity, `kernel="null"`.
`null_comparison` is the one-line version of that protocol.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# §5.2 experimental metrics
# --------------------------------------------------------------------------


def bimodality_coefficient(x: np.ndarray) -> float:
    """Polarization primitive: > 5/9 (~0.555, the value for a uniform
    distribution) suggests bimodality over unimodality. Pearson kurtosis
    (includes the +3), matching the standard `(skew^2 + 1) / kurtosis` form.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return float("nan")
    g = stats.skew(x)
    k = stats.kurtosis(x, fisher=False)
    return float((g**2 + 1) / k)


def cluster_stance_centroids(stance: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(lbl): stance[labels == lbl].mean(axis=0) for lbl in np.unique(labels)}


def cluster_centroid_distance(stance: np.ndarray, labels: np.ndarray) -> float:
    """Mean pairwise distance between cluster stance centroids — the
    "distance between cluster stance centroids" half of §5.2's polarization
    metric, tracked over time by calling this once per snapshot.
    """
    centroids = list(cluster_stance_centroids(stance, labels).values())
    if len(centroids) < 2:
        return 0.0
    dists = [np.linalg.norm(a - b) for i, a in enumerate(centroids) for b in centroids[i + 1 :]]
    return float(np.mean(dists))


def echo_chamber_index(own_stance: np.ndarray, consumed_stance: np.ndarray, user_id: np.ndarray, delta: float) -> np.ndarray:
    """Fraction of each user's consumed stance mass within `delta` of their
    own stance (spec §5.2). `own_stance` is (N, D); `consumed_stance` is
    (M, D), one row per (user, exposed-post) pair; `user_id` (M,) says whose
    exposure each row belongs to. Returns one share per user with >=1
    exposure (nan for users absent from `user_id`).
    """
    n = own_stance.shape[0]
    dist = np.linalg.norm(consumed_stance - own_stance[user_id], axis=1)
    within = (dist <= delta).astype(float)

    total = np.zeros(n)
    hits = np.zeros(n)
    np.add.at(total, user_id, 1.0)
    np.add.at(hits, user_id, within)

    share = np.full(n, np.nan)
    seen = total > 0
    share[seen] = hits[seen] / total[seen]
    return share


def attention_inequality(engagement_count: np.ndarray) -> tuple[float, float]:
    """(Gini, top-1%-share) of engagement across posts (spec §5.2)."""
    from discourse_lab.measures import attention_gini

    x = np.asarray(engagement_count, dtype=float)
    gini = attention_gini(x)
    if len(x) == 0 or x.sum() == 0:
        return gini, 0.0
    top_n = max(1, int(np.ceil(len(x) * 0.01)))
    top1_share = float(np.sort(x)[::-1][:top_n].sum() / x.sum())
    return gini, top1_share


def quality_attention_correlation(quality: np.ndarray, engagement_count: np.ndarray) -> float:
    """Spearman(quality_p, engagement) — under `kernel="null"` or
    `kernel="bandwagon"` this should sit near zero; verify it does (§5.2).
    """
    if len(quality) < 2:
        return float("nan")
    rho, _ = stats.spearmanr(quality, engagement_count)
    return float(rho)


def style_convergence(expression_snapshots: list[np.ndarray]) -> np.ndarray:
    """Trace of the expression-block covariance at each snapshot (spec
    §5.2): a shrinking trace over time is style convergence.
    """
    return np.array([np.trace(np.cov(snap, rowvar=False)) for snap in expression_snapshots])


def drift_magnitude(x_t: np.ndarray, x_0: np.ndarray, blocks: dict[str, list[int]]) -> dict[str, float]:
    """`||X_t - X_0||` by trait block (spec §5.2). `blocks` maps a block name
    to the column indices of `x_t`/`x_0` it covers.
    """
    return {name: float(np.linalg.norm(x_t[:, cols] - x_0[:, cols])) for name, cols in blocks.items()}


def null_comparison(effect: float, null_effect: float) -> float:
    """The one-line version of the §5.3 protocol: an effect only counts once
    it is compared against its matched `kernel="null"` run.
    """
    return effect - null_effect


# --------------------------------------------------------------------------
# §5.1 stylized facts
# --------------------------------------------------------------------------

STYLIZED_FACT_RANGES: dict[str, tuple[float, float]] = {
    "attention_gini": (0.8, 0.95),
    "posting_volume_gini": (0.7, 0.9),
    "reciprocity": (0.2, 0.4),
    "cascade_singleton_share": (0.9, 1.0),
    "thread_depth_mean": (1.5, 3.0),
}


def stylized_facts_report(
    attention_gini: float | None = None,
    posting_volume_gini: float | None = None,
    reciprocity: float | None = None,
    cascade_singleton_share: float | None = None,
    thread_depth_mean: float | None = None,
) -> dict[str, dict]:
    """Check whichever facts are supplied against spec §5.1's target ranges.
    The simulation is calibrated when these emerge without being imposed —
    do not proceed past this step until they hold (dev §6 step 9).
    """
    values = {
        "attention_gini": attention_gini,
        "posting_volume_gini": posting_volume_gini,
        "reciprocity": reciprocity,
        "cascade_singleton_share": cascade_singleton_share,
        "thread_depth_mean": thread_depth_mean,
    }
    report = {}
    for name, value in values.items():
        if value is None:
            continue
        lo, hi = STYLIZED_FACT_RANGES[name]
        report[name] = {"value": value, "target": (lo, hi), "in_range": lo <= value <= hi}
    return report
