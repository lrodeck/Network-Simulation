"""Trait -> expression map (spec §2.4): `d_p = A @ x_u + B @ s(t) + C @ onehot(topic_p) + eps_d`.

`A` is the single most important authored object in the system — it encodes
claims like "high neuroticism raises arousal" or "low agreeableness raises
provocativeness". Authored as a sparse, commented, named-entry table, never a
dense matrix, so every claim it makes is legible and individually editable.

Post dims are stored unconstrained, same discipline as user traits (spec
§1.1): a link per dim enforces the constraint on read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discourse_lab.population.links import to_used

POST_DIMS = ("arousal", "valence", "provocativeness", "novelty", "specificity", "quality", "length")

POST_DIM_LINKS = {
    "arousal": "logit",
    "valence": "tanh",
    "provocativeness": "logit",
    "novelty": "logit",
    "specificity": "logit",
    "quality": "logit",
    "length": "log",
}

# A: (post_dim, trait, weight) — the trait->expression map. Every entry is a
# specific, individually-arguable claim; unlisted (dim, trait) pairs are zero.
DEFAULT_A: tuple[tuple[str, str, float], ...] = (
    ("arousal", "neuroticism", 0.6),          # high neuroticism raises arousal
    ("arousal", "extraversion", 0.3),
    ("valence", "neuroticism", -0.4),         # neuroticism skews expression negative
    ("valence", "agreeableness", 0.3),
    ("provocativeness", "agreeableness", -0.6),   # low agreeableness raises provocativeness
    ("provocativeness", "contrarianism", 0.7),
    ("novelty", "openness", 0.5),
    ("specificity", "conscientiousness", 0.5),
    ("specificity", "verbosity", 0.2),
    ("quality", "conscientiousness", 0.4),
    ("quality", "credulity", -0.3),            # credulous authors post lower-merit claims
    ("length", "verbosity", 0.8),
    ("length", "formality", 0.3),
)

# B: (post_dim, source) — pressure from the *overall* level of discourse
# attention (sum over topics of s(t)), independent of which topic this post is in.
DEFAULT_B: tuple[tuple[str, float], ...] = (
    ("arousal", 0.5),     # a hot news cycle raises arousal across the board
    ("novelty", -0.3),    # heavily-attended topics feel less novel to write about
)

# C: (post_dim, topic, weight) — per-topic fixed baseline offsets. Sparse:
# most topics carry no offset for most dims.
DEFAULT_C: tuple[tuple[str, int, float], ...] = ()


@dataclass
class ExpressionMap:
    trait_names: list[str]
    n_topics: int
    A: np.ndarray  # (p, n)
    B: np.ndarray  # (p,)
    C: np.ndarray  # (p, K)

    @classmethod
    def build(
        cls,
        trait_names: list[str],
        n_topics: int,
        entries_a: tuple[tuple[str, str, float], ...] = DEFAULT_A,
        entries_b: tuple[tuple[str, float], ...] = DEFAULT_B,
        entries_c: tuple[tuple[str, int, float], ...] = DEFAULT_C,
    ) -> "ExpressionMap":
        p = len(POST_DIMS)
        dim_idx = {d: i for i, d in enumerate(POST_DIMS)}
        trait_idx = {t: i for i, t in enumerate(trait_names)}

        A = np.zeros((p, len(trait_names)))
        for dim, trait, w in entries_a:
            j = trait_idx.get(trait)
            if j is not None:
                A[dim_idx[dim], j] = w

        B = np.zeros(p)
        for dim, w in entries_b:
            B[dim_idx[dim]] = w

        C = np.zeros((p, n_topics))
        for dim, topic, w in entries_c:
            if 0 <= topic < n_topics:
                C[dim_idx[dim], topic] = w

        return cls(trait_names=trait_names, n_topics=n_topics, A=A, B=B, C=C)

    def generate(
        self,
        X_stored: np.ndarray,
        topic_p: np.ndarray,
        s_t: np.ndarray,
        rng: np.random.Generator,
        noise_sigma: float = 0.15,
    ) -> dict[str, np.ndarray]:
        """`d_p = A @ x_u + B * sum(s(t)) + C[:, topic_p] + eps_d`, over
        unconstrained (stored-space) traits `x_u` — the same space drift
        operates on — converted to used space per §1.1's discipline.
        """
        n = X_stored.shape[0]
        stored = X_stored @ self.A.T                # (n, p)
        stored += self.B[None, :] * s_t.sum()       # broadcast global attention pressure
        stored += self.C[:, topic_p].T              # (n, p) topic baseline per post
        stored += rng.normal(0, noise_sigma, size=stored.shape)

        used = {}
        for i, dim in enumerate(POST_DIMS):
            used[dim] = to_used(stored[:, i], POST_DIM_LINKS[dim])
        return used
