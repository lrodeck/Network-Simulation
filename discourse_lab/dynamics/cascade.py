"""Frontier cascades (spec §2.7): a repost or quote creates a derived post
inheriting `root_p`, with dims perturbed (quotes shift stance toward the
quoter, reposts do not). It re-enters exposure with the reposter as author,
visible with multiplier `rho ** depth` (applied in `exposure.attention`).

    R = E[# reposts per exposure] * E[audience per repost]

Calibrated so `E[R] < 1` (cascades usually die) but `Var[R]` large enough
that the tail crosses 1 — subcritical-with-heavy-tail reproduces observed
cascade size distributions. `r_eff` below is the per-tick diagnostic; log it
and warn rather than silently truncating if it sits above 1.

Hard caps (`max_cascade_depth`, `max_cascade_size`) trip a warning instead of
silently truncating, per the spec's explicit instruction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from discourse_lab.dynamics.expression import ExpressionMap
from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.network import Graph
from discourse_lab.population import Population

CASCADE_ACTIONS = ("repost", "quote")


@dataclass
class CascadeState:
    """Tracks total posts per root tree, across ticks, for `max_cascade_size`."""

    size_by_root: dict[int, int] = field(default_factory=dict)


def derive_posts(
    actions: np.ndarray,
    exposure_post_idx: np.ndarray,
    exposure_user_id: np.ndarray,
    posts: PostBatch,
    pop: Population,
    expression: ExpressionMap,
    s_t: np.ndarray,
    sigma_t: np.ndarray,
    rng: np.random.Generator,
    cascade_state: CascadeState,
    max_depth: int,
    max_size: int,
    start_id: int,
    t: int,
    quote_stance_shift: float = 0.4,
) -> tuple[PostBatch | None, list[str]]:
    """Turn this tick's repost/quote actions into a new `PostBatch` of
    derived posts. Returns `(None, warnings)` if nothing survived depth/size
    caps. `posts` must be the batch the exposures reference.
    """
    warnings_out: list[str] = []
    mask = np.isin(actions, CASCADE_ACTIONS)
    if not mask.any():
        return None, warnings_out

    reposter = exposure_user_id[mask]
    orig_idx = exposure_post_idx[mask]
    kind = actions[mask]
    parent_id = posts.id[orig_idx]
    root_id = posts.root[orig_idx]
    depth = posts.depth[orig_idx] + 1

    within_depth = depth <= max_depth
    if not within_depth.all():
        warnings_out.append(
            f"cascade: dropped {int((~within_depth).sum())} derived post(s) past max_cascade_depth={max_depth}"
        )
    reposter, orig_idx, kind = reposter[within_depth], orig_idx[within_depth], kind[within_depth]
    parent_id, root_id, depth = parent_id[within_depth], root_id[within_depth], depth[within_depth]

    allowed = np.ones(len(root_id), dtype=bool)
    for root in np.unique(root_id):
        idxs = np.flatnonzero(root_id == root)
        current = cascade_state.size_by_root.get(int(root), 1)  # the root post itself counts as 1
        capacity = max(max_size - current, 0)
        if len(idxs) > capacity:
            allowed[idxs[capacity:]] = False
            warnings_out.append(
                f"cascade: root {int(root)} hit max_cascade_size={max_size}, "
                f"dropped {len(idxs) - capacity} derived post(s)"
            )
        cascade_state.size_by_root[int(root)] = current + min(len(idxs), capacity)

    if not allowed.any():
        return None, warnings_out
    reposter, orig_idx, kind = reposter[allowed], orig_idx[allowed], kind[allowed]
    parent_id, root_id, depth = parent_id[allowed], root_id[allowed], depth[allowed]

    topic_p = posts.topic[orig_idx]
    stance_orig = posts.stance[orig_idx]

    names = pop.trait_names
    stance_cols = [i for i, n in enumerate(names) if n.startswith("stance_")]
    stance_quoter = pop.X_used[reposter][:, stance_cols]

    is_quote = kind == "quote"
    stance_new = stance_orig.copy()
    stance_new[is_quote] = (
        (1 - quote_stance_shift) * stance_orig[is_quote] + quote_stance_shift * stance_quoter[is_quote]
    )

    m = len(reposter)
    d_fields = {
        "arousal": posts.arousal[orig_idx].copy(),
        "valence": posts.valence[orig_idx].copy(),
        "provocativeness": posts.provocativeness[orig_idx].copy(),
        "novelty": posts.novelty[orig_idx].copy(),
        "specificity": posts.specificity[orig_idx].copy(),
        "quality": posts.quality[orig_idx].copy(),
        "length": posts.length[orig_idx].copy(),
    }
    if is_quote.any():
        # quotes re-run the expression map for the quoter's own perturbation
        # on top of the original content; reposts leave dims untouched.
        d_quote = expression.generate(pop.X_stored[reposter[is_quote]], topic_p[is_quote], s_t, rng)
        for key in d_fields:
            d_fields[key][is_quote] = 0.5 * d_fields[key][is_quote] + 0.5 * d_quote[key]

    ids = np.arange(start_id, start_id + m)
    new_posts = PostBatch(
        author=reposter,
        topic=topic_p,
        stance=stance_new,
        arousal=d_fields["arousal"],
        valence=d_fields["valence"],
        provocativeness=d_fields["provocativeness"],
        novelty=d_fields["novelty"],
        specificity=d_fields["specificity"],
        quality=d_fields["quality"],
        length=d_fields["length"],
        id=ids,
        t=np.full(m, t),
        parent=parent_id,
        root=root_id,
        depth=depth,
        kind=kind,
        engagement_count=np.zeros(m, dtype=np.int64),
    )
    return new_posts, warnings_out


def follower_counts(graph: Graph) -> np.ndarray:
    """Followers-of count per user, from the CSC column pointers."""
    return np.diff(graph.csc.indptr)


def r_eff(actions: np.ndarray, n_exposures: int, graph: Graph, reposter_ids: np.ndarray | None = None) -> float:
    """`R = E[# reposts per exposure] * E[audience per repost]` (spec §2.7).
    `reposter_ids` restricts the audience estimate to this tick's actual
    reposters; omit to use the population mean follower count.
    """
    n_reposts = int(np.isin(actions, CASCADE_ACTIONS).sum())
    p_repost = n_reposts / max(n_exposures, 1)

    counts = follower_counts(graph)
    avg_audience = counts[reposter_ids].mean() if reposter_ids is not None and len(reposter_ids) > 0 else counts.mean()

    return float(p_repost * avg_audience)


def check_r_eff(value: float) -> None:
    if value > 1.0:
        warnings.warn(
            f"R_eff={value:.3f} > 1: cascades are supercritical this tick, the run will saturate", stacklevel=2
        )
