"""Rankers (spec §2.5b): `rank: (u, I_u, X, D_p, state) -> ordered list`,
implemented here as a per-candidate score — higher sorts first. Registered by
name so `cfg.dynamics.ranker` picks the theory of the feed.
"""

from __future__ import annotations

import numpy as np

from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.exposure.inbox import CandidatePairs
from discourse_lab.population import Population
from discourse_lab.registry import register


@register("ranker", "chronological")
def chronological(pairs: CandidatePairs, posts: PostBatch, pop: Population, rng: np.random.Generator) -> np.ndarray:
    return posts.t[pairs.post_idx].astype(float)


@register("ranker", "random")
def random_ranker(pairs: CandidatePairs, posts: PostBatch, pop: Population, rng: np.random.Generator) -> np.ndarray:
    return rng.random(len(pairs))


@register("ranker", "popularity")
def popularity(pairs: CandidatePairs, posts: PostBatch, pop: Population, rng: np.random.Generator) -> np.ndarray:
    return posts.engagement_count[pairs.post_idx].astype(float)


@register("ranker", "affinity")
def affinity(pairs: CandidatePairs, posts: PostBatch, pop: Population, rng: np.random.Generator) -> np.ndarray:
    names = pop.trait_names
    topic_cols = [i for i, n in enumerate(names) if n.startswith("topic_affinity_")]
    stance_cols = [i for i, n in enumerate(names) if n.startswith("stance_")]

    a_u = pop.X_used[pairs.user_id][:, topic_cols]
    topic_p = posts.topic[pairs.post_idx]
    topic_affinity_score = a_u[np.arange(len(pairs)), topic_p]

    s_u = pop.X_used[pairs.user_id][:, stance_cols]
    s_p = posts.stance[pairs.post_idx]
    agreement = -np.linalg.norm(s_u - s_p, axis=1)

    return topic_affinity_score + agreement


@register("ranker", "engagement_optimized")
def engagement_optimized(pairs: CandidatePairs, posts: PostBatch, pop: Population, rng: np.random.Generator) -> np.ndarray:
    """Sort by a cheap proxy for P(engage): affinity/agreement plus arousal
    and social proof — a stand-in for a trained propensity model.
    """
    aff_score = affinity(pairs, posts, pop, rng)
    arousal = posts.arousal[pairs.post_idx]
    social_proof = np.log1p(posts.engagement_count[pairs.post_idx])
    return aff_score + arousal + social_proof
