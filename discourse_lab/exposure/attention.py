"""Attention budget (spec §2.5c): the finite attention budget is
load-bearing. Without it exposure grows unbounded and virality has nothing to
compete against — attention is the scarce resource the whole system fights
over.

    B_u ~ Poisson(b * activity_u)
    P(see item at rank r) = exp(-r / tau_pos)

**These two caps compose, and the softer one almost always binds first.**
Position decay alone passes about `tau_pos` items in expectation (6 at the
default), so a budget set anywhere above that is not a constraint at all.
Measured, on a user with 200 candidates and activity 1.0, the fraction of
decay-survivors that the budget additionally removes:

    b =   3   ->  60%        b =  30   ->   1.2%
    b =  10   ->  25%        b =  60   ->   0.0%
    b =  15   ->  10%        b = 120   ->   0.0%

At the shipped default of b = 30 the budget is very nearly decorative, and any
sweep across 30/60/120 compares three identical platforms. This cost a 10-seed
intervention sweep, which reported `attention_budget` as inert across all 13
outcome columns; it was swept entirely inside its dead zone. Vary `tau_position`
to ration attention, or take `attention_budget` below ~15 where it starts to
bite. `tests/test_attention_budget_binds.py` pins the interaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discourse_lab.exposure.inbox import CandidatePairs


@dataclass
class Exposures:
    post_idx: np.ndarray
    user_id: np.ndarray
    rank: np.ndarray
    is_follower: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.post_idx)


def select_exposures(
    pairs: CandidatePairs,
    scores: np.ndarray,
    activity: np.ndarray,
    attention_budget: float,
    tau_position: float,
    rng: np.random.Generator,
    cascade_depth: np.ndarray | None = None,
    cascade_rho: float = 1.0,
) -> Exposures:
    """Rank each user's candidates by score, cap at their Poisson attention
    budget `B_u`, then thin by position-decay visibility.

    `cascade_depth` (aligned with `pairs`, spec §2.7) additionally multiplies
    visibility by `cascade_rho ** depth`: a derived (repost/quote) post's
    reach shrinks geometrically with how far it is from its root, on top of
    ordinary position decay. Omit it (the default) for root-only posts, where
    depth is always 0 and the multiplier is exactly 1.
    """
    if len(pairs) == 0:
        empty = np.empty(0, dtype=np.int64)
        return Exposures(post_idx=empty, user_id=empty, rank=empty, is_follower=np.empty(0, dtype=bool))

    order = np.lexsort((-scores, pairs.user_id))  # primary key: user_id asc; secondary: score desc
    sorted_user = pairs.user_id[order]
    sorted_post = pairs.post_idx[order]
    sorted_is_follower = pairs.is_follower[order]

    unique_users, start_idx, counts = np.unique(sorted_user, return_index=True, return_counts=True)
    rank = np.arange(len(sorted_user)) - np.repeat(start_idx, counts)

    budgets = rng.poisson(attention_budget * activity[unique_users])
    budget_per_row = np.repeat(budgets, counts)

    within_budget = rank < budget_per_row
    visibility = np.exp(-rank / tau_position)
    if cascade_depth is not None:
        visibility = visibility * (cascade_rho ** cascade_depth[order])
    seen = rng.random(len(sorted_user)) < visibility
    keep = within_budget & seen

    return Exposures(
        post_idx=sorted_post[keep], user_id=sorted_user[keep], rank=rank[keep], is_follower=sorted_is_follower[keep]
    )
