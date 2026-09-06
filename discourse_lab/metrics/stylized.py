"""The spec §5.1 stylized facts, computed from a run's persisted tables.

Every function here takes plain arrays (or the columns of `posts.parquet` /
`engagements.parquet`) rather than a `RunHandle`, so they are testable on
hand-built data and reusable on in-memory batches from `run_iter`.

Two interpretation decisions are baked in here, both because the spec's own
table is ambiguous or self-contradictory. They are called out in the
docstrings rather than buried: see `thread_depth_mean` and
`posting_volume_gini`.
"""

from __future__ import annotations

import numpy as np

from discourse_lab.measures import gini

HOSTILE_ACTIONS = ("reply", "report", "quote")


# --------------------------------------------------------------------------
# cascades
# --------------------------------------------------------------------------


def cascade_sizes(root: np.ndarray) -> np.ndarray:
    """Number of posts per cascade, one entry per distinct root."""
    root = np.asarray(root)
    if len(root) == 0:
        return np.array([], dtype=np.int64)
    _, counts = np.unique(root, return_counts=True)
    return counts


def cascade_singleton_share(root: np.ndarray) -> float:
    """Spec §5.1: >90% of cascades should be size 1."""
    sizes = cascade_sizes(root)
    if len(sizes) == 0:
        return float("nan")
    return float((sizes == 1).mean())


def thread_depths(root: np.ndarray, depth: np.ndarray, min_size: int = 2) -> np.ndarray:
    """Maximum depth reached per cascade, over cascades of at least
    `min_size` posts.

    **Interpretation.** The spec's §5.1 table asks for ">90% of cascades size
    1" and "thread depth mean 1.5-3" simultaneously. Those cannot both hold
    over all cascades: if nine in ten are a lone post at depth 0, the mean
    depth is ~0.05, and even counting levels (depth+1) it is ~1.05. Depth is
    therefore measured **only over cascades that actually branched**
    (`min_size=2`), where depth is 0-indexed from the root — so a cascade with
    one repost has depth 1, and the spec's 1.5-3 band is reachable and
    meaningful. Reported unconditionally it would fail by construction and
    look like a calibration failure that is really a definition mismatch.
    """
    root = np.asarray(root)
    depth = np.asarray(depth)
    if len(root) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(root, kind="stable")
    root_sorted, depth_sorted = root[order], depth[order]
    uniq, starts, counts = np.unique(root_sorted, return_index=True, return_counts=True)

    max_depth = np.maximum.reduceat(depth_sorted, starts)
    return max_depth[counts >= min_size]


def thread_depth_mean(root: np.ndarray, depth: np.ndarray, min_size: int = 2) -> float:
    depths = thread_depths(root, depth, min_size=min_size)
    if len(depths) == 0:
        return float("nan")
    return float(depths.mean())


# --------------------------------------------------------------------------
# inequality
# --------------------------------------------------------------------------


def posting_volume_gini(author: np.ndarray, n_users: int) -> float:
    """Gini of posts-per-user across the **whole population**.

    **Interpretation.** `n_users` is required, not inferred from the authors
    present, because users who never posted are the bulk of the distribution
    — the population is mostly lurkers by construction (the `lurker`
    archetype alone is 55% of the default mixture). Computing this over
    posting users only would silently drop them and understate the
    inequality the spec's 0.7-0.9 band is describing.

    Callers should pass root posts only (`kind == "post"`); cascade-derived
    reposts and quotes are a different behaviour and the metric column
    `n_posts` excludes them too.
    """
    counts = np.bincount(np.asarray(author, dtype=np.int64), minlength=n_users)
    return gini(counts)


def lorenz_curve(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(population share, value share)`, both starting at (0, 0) — the
    cumulative curve whose gap from the diagonal is the Gini.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or x.sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    ordered = np.sort(x)
    value_share = np.concatenate([[0.0], np.cumsum(ordered) / ordered.sum()])
    population_share = np.linspace(0.0, 1.0, len(value_share))
    return population_share, value_share


# --------------------------------------------------------------------------
# cross-cluster contact
# --------------------------------------------------------------------------


def stance_clusters(stance: np.ndarray) -> np.ndarray:
    """Two ideological camps: the sign of each user's position on the dominant
    axis of stance variation (the first principal component).

    **Interpretation.** Spec §5.1's "inter-cluster interaction rate" needs a
    notion of cluster, and archetype labels are the wrong one: archetypes are
    behavioural roles (lurker, poster, institution) the dynamics never read,
    so a lurker engaging a poster is ordinary traffic, not a boundary
    crossing. Ideological position is what the fact is about — "low contact,
    high hostility given contact" is a claim about opposing camps.

    Two camps, not 2^D. A median split per axis was the first implementation
    and it does not survive D > 1: at the spec's D = 3 it makes 8 camps, so
    even a perfectly homophilous population crosses a boundary on 87.5% of
    random contacts and the measured rate (0.63) sits *below* chance while
    reading as a catastrophic failure. Projecting onto the first principal
    component instead measures the polarisation axis the population actually
    has, at any D, and keeps the rate comparable across scenarios. This also
    matches spec §7.5's expectation that correlated axes collapse toward a
    single dominant dimension.
    """
    stance = np.atleast_2d(np.asarray(stance, dtype=float))
    if stance.shape[0] == 1 and stance.shape[1] > 1:
        stance = stance.T
    centred = stance - stance.mean(axis=0)
    if centred.shape[1] == 1:
        projection = centred[:, 0]
    else:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        projection = centred @ vt[0]
    return (projection > np.median(projection)).astype(np.int64)


def inter_cluster_interaction(
    engaging_user: np.ndarray,
    post_author: np.ndarray,
    action: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Spec §5.1's last row: "inter-cluster interaction rate low, but
    hostility rate high given contact".

    Returns `(cross_cluster_rate, hostility_given_contact)`. `labels` should
    be ideological clusters — see `stance_clusters` for why archetype labels
    are not the right input. Hostility is the share of `reply`/`report`/
    `quote` among engagements that crossed a boundary: likes and reposts are
    the affiliative actions, replies, quotes and reports the contentious ones.
    """
    engaging_user = np.asarray(engaging_user, dtype=np.int64)
    post_author = np.asarray(post_author, dtype=np.int64)
    action = np.asarray(action)
    if len(engaging_user) == 0:
        return float("nan"), float("nan")

    crossed = labels[engaging_user] != labels[post_author]
    cross_rate = float(crossed.mean())

    if not crossed.any():
        return cross_rate, float("nan")
    hostile = np.isin(action[crossed], HOSTILE_ACTIONS)
    return cross_rate, float(hostile.mean())
