"""Frontier cascades (spec §2.7): a repost, quote or reply creates a derived
post inheriting `root_p`, with dims perturbed. It re-enters exposure with the
deriving user as author, visible with multiplier `rho ** depth` (applied in
`exposure.attention`).

The three derivation kinds differ in how far the new post's stance moves from
its parent's toward the deriving user's own:

    repost  0.0   verbatim amplification, dims untouched
    quote   0.4   the quoter frames someone else's post
    reply   1.0   the replier's own words, on the parent's topic

Reply spawning is why thread depth is a real measurement rather than a
degenerate one. Reposts fan out wide and shallow; on real platforms depth
comes from reply chains. `reply` was previously counted as an engagement and
fed the discourse-state update but never created a post, so `depth` could
only grow through repost/quote and the spec §5.1 depth target was
unreachable by construction.

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

CASCADE_ACTIONS = ("repost", "quote", "reply")

# How far a derived post's stance moves from its parent toward the deriving
# user's own stance. See the module docstring.
STANCE_SHIFT = {"repost": 0.0, "quote": 0.4, "reply": 1.0}

# Actions that fan a post out to a *new* audience, which is what the
# branching factor R in `r_eff` measures. A reply is visible to the replier's
# followers too, but it is a response rather than an amplification, and spec
# §2.7 defines R over reposts.
BRANCHING_ACTIONS = ("repost", "quote")


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

    # per-row stance shift toward the deriving user (0 for reposts, so they
    # stay verbatim); `quote_stance_shift` overrides the quote row for
    # callers that tune it
    shift = np.array([STANCE_SHIFT[k] for k in kind])
    shift[kind == "quote"] = quote_stance_shift
    stance_new = (1 - shift)[:, None] * stance_orig + shift[:, None] * stance_quoter

    # reposts carry the original content; quotes and replies re-run the
    # expression map for the deriving user's own perturbation
    regenerates = kind != "repost"

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
    if regenerates.any():
        d_new = expression.generate(pop.X_stored[reposter[regenerates]], topic_p[regenerates], s_t, rng)
        # a quote is half the original and half the quoter; a reply is the
        # replier's own content, only the topic is inherited
        blend = np.where(kind[regenerates] == "reply", 1.0, 0.5)
        for key in d_fields:
            d_fields[key][regenerates] = (1 - blend) * d_fields[key][regenerates] + blend * d_new[key]

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
    n_reposts = int(np.isin(actions, BRANCHING_ACTIONS).sum())
    p_repost = n_reposts / max(n_exposures, 1)

    counts = follower_counts(graph)
    avg_audience = counts[reposter_ids].mean() if reposter_ids is not None and len(reposter_ids) > 0 else counts.mean()

    return float(p_repost * avg_audience)


def check_r_eff(value: float) -> None:
    if value > 1.0:
        warnings.warn(
            f"R_eff={value:.3f} > 1: cascades are supercritical this tick, the run will saturate", stacklevel=2
        )
