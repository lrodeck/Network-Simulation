"""Attention budget (spec §2.5c): the finite attention budget is
load-bearing. Without it exposure grows unbounded and virality has nothing to
compete against — attention is the scarce resource the whole system fights
over.

    B_u ~ Poisson(b * activity_u)
    P(see item at rank r) = exp(-r / tau_pos)
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

    def __len__(self) -> int:
        return len(self.post_idx)


def select_exposures(
    pairs: CandidatePairs,
    scores: np.ndarray,
    activity: np.ndarray,
    attention_budget: float,
    tau_position: float,
    rng: np.random.Generator,
) -> Exposures:
    """Rank each user's candidates by score, cap at their Poisson attention
    budget `B_u`, then thin by position-decay visibility.
    """
    if len(pairs) == 0:
        empty = np.empty(0, dtype=np.int64)
        return Exposures(post_idx=empty, user_id=empty, rank=empty)

    order = np.lexsort((-scores, pairs.user_id))  # primary key: user_id asc; secondary: score desc
    sorted_user = pairs.user_id[order]
    sorted_post = pairs.post_idx[order]

    unique_users, start_idx, counts = np.unique(sorted_user, return_index=True, return_counts=True)
    rank = np.arange(len(sorted_user)) - np.repeat(start_idx, counts)

    budgets = rng.poisson(attention_budget * activity[unique_users])
    budget_per_row = np.repeat(budgets, counts)

    within_budget = rank < budget_per_row
    visibility = np.exp(-rank / tau_position)
    seen = rng.random(len(sorted_user)) < visibility
    keep = within_budget & seen

    return Exposures(post_idx=sorted_post[keep], user_id=sorted_user[keep], rank=rank[keep])
