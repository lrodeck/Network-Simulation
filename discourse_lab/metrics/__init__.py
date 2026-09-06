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

from discourse_lab.metrics.powerlaw import PowerLawFit, ccdf, powerlaw_alpha, powerlaw_fit
from discourse_lab.metrics.stylized import (
    cascade_singleton_share,
    cascade_sizes,
    inter_cluster_interaction,
    lorenz_curve,
    posting_volume_gini,
    stance_clusters,
    thread_depth_mean,
    thread_depths,
)

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

# All 8 rows of the spec §5.1 table. The three that had no encoded range
# before (`engagement_alpha`, `clustering_ratio`, `inter_cluster_rate`) are
# read off the table's prose: alpha in [2,3]; clustering "≫ random graph of
# same degree", taken as at least 3x; inter-cluster interaction "low", taken
# as under a third of engagements.
STYLIZED_FACT_RANGES: dict[str, tuple[float, float]] = {
    "engagement_alpha": (2.0, 3.0),
    "cascade_singleton_share": (0.9, 1.0),
    "thread_depth_mean": (1.5, 3.0),
    "attention_gini": (0.8, 0.95),
    "posting_volume_gini": (0.7, 0.9),
    "reciprocity": (0.2, 0.4),
    "clustering_ratio": (3.0, np.inf),
    "inter_cluster_rate": (0.0, 0.33),
}

STYLIZED_FACT_LABELS: dict[str, str] = {
    "engagement_alpha": "Engagement per post (power-law alpha)",
    "cascade_singleton_share": "Cascade size (share of singletons)",
    "thread_depth_mean": "Thread depth (mean, branched cascades)",
    "attention_gini": "Attention Gini (lifetime, per post)",
    "posting_volume_gini": "Posting volume Gini",
    "reciprocity": "Reciprocity",
    "clustering_ratio": "Clustering vs degree-matched null",
    "inter_cluster_rate": "Inter-cluster interaction rate",
}


def stylized_facts_report(**values: float | None) -> dict[str, dict]:
    """Check whichever facts are supplied against spec §5.1's target ranges.
    The simulation is calibrated when these emerge without being imposed —
    do not proceed past this step until they hold (dev §6 step 9).

    Unsupplied facts are omitted rather than reported as failures, so a
    partial run (no posts persisted, say) still produces an honest table of
    what it could actually measure.
    """
    unknown = set(values) - set(STYLIZED_FACT_RANGES)
    if unknown:
        raise ValueError(f"unknown stylized fact(s): {sorted(unknown)}")

    report = {}
    for name in STYLIZED_FACT_RANGES:  # spec table order, not kwargs order
        value = values.get(name)
        if value is None:
            continue
        lo, hi = STYLIZED_FACT_RANGES[name]
        report[name] = {
            "label": STYLIZED_FACT_LABELS[name],
            "value": float(value),
            "target": (lo, hi),
            "in_range": bool(lo <= value <= hi) if np.isfinite(value) else False,
        }
    return report


def stylized_facts_from_run(handle, graph=None, pop=None, rng=None) -> dict[str, dict]:
    """Compute every §5.1 fact this run has the data for, and report it.

    `handle` is a `RunHandle`. Facts needing per-post data are skipped unless
    the run was written with `persist=("posts",)`; the cross-cluster fact also
    needs `"engagements"` plus `pop` for archetype labels; the graph facts
    need `graph`. Whatever is missing is simply absent from the report rather
    than reported as a failure — see `stylized_facts_report`.

    Note the attention Gini here is **lifetime engagement per post**, not the
    rolling per-tick `attention_gini` column in `metrics.parquet`. The two
    measure different things and will not agree; any table built from this
    should say which it shows.
    """
    from discourse_lab.measures import gini
    from discourse_lab.network.measures import clustering_vs_random
    from discourse_lab.network.measures import reciprocity as measure_reciprocity

    rng = rng if rng is not None else np.random.default_rng(0)
    facts: dict[str, float] = {}
    engagement_fit = None

    if handle.has_posts:
        posts = handle.posts()
        engagement = posts["engagement_count"].to_numpy()
        engagement_fit = powerlaw_fit(engagement)
        facts["engagement_alpha"] = engagement_fit.alpha
        facts["attention_gini"] = gini(engagement)

        root = posts["root"].to_numpy()
        facts["cascade_singleton_share"] = cascade_singleton_share(root)
        facts["thread_depth_mean"] = thread_depth_mean(root, posts["depth"].to_numpy())

        roots_only = posts.filter(posts["kind"] == "post")
        n_users = int(handle.meta["config"]["population"]["n_users"])
        facts["posting_volume_gini"] = posting_volume_gini(roots_only["author"].to_numpy(), n_users)

    if graph is not None:
        facts["reciprocity"] = measure_reciprocity(graph.csr)
        facts["clustering_ratio"] = clustering_vs_random(graph.csr, rng)[2]

    if handle.has_posts and handle.has_engagements and pop is not None:
        posts = handle.posts()
        engagements = handle.engagements()
        author_of_post = dict(zip(posts["id"].to_list(), posts["author"].to_list()))
        post_ids = engagements["post"].to_numpy()
        authors = np.array([author_of_post.get(int(p), -1) for p in post_ids])
        known = authors >= 0
        if known.any():
            # ideological camps, not archetypes — see stylized.stance_clusters
            stance_cols = [i for i, n in enumerate(pop.trait_names) if n.startswith("stance_")]
            labels = stance_clusters(pop.X_used[:, stance_cols])
            rate, hostility = inter_cluster_interaction(
                engagements["user"].to_numpy()[known],
                authors[known],
                engagements["action"].to_numpy()[known],
                labels,
            )
            facts["inter_cluster_rate"] = rate
            facts["_hostility_given_contact"] = hostility

    hostility = facts.pop("_hostility_given_contact", None)
    report = stylized_facts_report(**facts)

    if engagement_fit is not None and "engagement_alpha" in report:
        # spec §5.1 asks for alpha in [2, 3], which presumes the distribution
        # is a power law at all. It is not: alpha climbs monotonically with
        # x_min (1.54 at 2, 5.66 at 300), so grading a single value against a
        # range reports on where x_min landed rather than on the model. The
        # row carries the diagnostic and is ungraded when the tail is curved.
        entry = report["engagement_alpha"]
        entry["alpha_spread"] = engagement_fit.alpha_spread
        entry["is_powerlaw"] = engagement_fit.is_powerlaw
        entry["xmin"] = engagement_fit.xmin
        entry["n_tail"] = engagement_fit.n_tail
        if not engagement_fit.is_powerlaw:
            entry["in_range"] = None
            entry["label"] += " — not a power law"
    if hostility is not None:
        # the other half of the fact: "hostility rate high given contact".
        # No range is quoted in the spec, so it is reported, not graded.
        report["hostility_given_contact"] = {
            "label": "Hostility given cross-cluster contact",
            "value": float(hostility),
            "target": None,
            "in_range": None,
        }
    return report
