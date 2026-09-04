"""Exposure and reaction (spec §2.5-2.6): candidate inbox, ranking, attention
budget, and the engagement kernel — the layer where platform design lives.
"""

from discourse_lab.exposure.attention import Exposures, select_exposures
from discourse_lab.exposure.inbox import CandidatePairs, candidate_inbox
from discourse_lab.exposure.kernel import ACTIONS, FEATURES, apply_kernel, compute_features, named_kernel
from discourse_lab.exposure import rankers  # noqa: F401  (registration side effects)
from discourse_lab.registry import get as _get


def rank_candidates(name: str, pairs: CandidatePairs, posts, pop, rng):
    return _get("ranker", name)(pairs, posts, pop, rng)


__all__ = [
    "CandidatePairs",
    "candidate_inbox",
    "Exposures",
    "select_exposures",
    "rank_candidates",
    "ACTIONS",
    "FEATURES",
    "compute_features",
    "apply_kernel",
    "named_kernel",
]
