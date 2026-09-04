"""Post dimension generation (spec §2.4).

    topic_p  ~ Categorical(softmax(a_u + eta * s(t)))
    stance_p = conviction_u * s_u + (1 - conviction_u) * sigma(t)[topic_p] + eps_s
    d_p      = A @ x_u + B @ s(t) + C @ onehot(topic_p) + eps_d

The stance line is the conformity mechanism: low-conviction users drift
toward whatever stance currently dominates their topic. `conviction = 1`
globally disables conformity as a control condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discourse_lab.dynamics.expression import ExpressionMap
from discourse_lab.population import Population


@dataclass
class PostBatch:
    author: np.ndarray
    topic: np.ndarray
    stance: np.ndarray             # (M, D)
    arousal: np.ndarray
    valence: np.ndarray
    provocativeness: np.ndarray
    novelty: np.ndarray
    specificity: np.ndarray
    quality: np.ndarray
    length: np.ndarray

    def __len__(self) -> int:
        return len(self.author)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sample_topics(rng: np.random.Generator, topic_affinity: np.ndarray, s_t: np.ndarray, eta: float) -> np.ndarray:
    logits = topic_affinity + eta * s_t[None, :]
    probs = _softmax(logits)
    cdf = np.cumsum(probs, axis=1)
    u = rng.random((probs.shape[0], 1))
    return (u < cdf).argmax(axis=1)


def sample_stance(
    rng: np.random.Generator,
    conviction: np.ndarray,
    stance_u: np.ndarray,
    topic_p: np.ndarray,
    sigma_t: np.ndarray,
    noise_sigma: float = 0.1,
) -> np.ndarray:
    dominant = sigma_t[topic_p]  # (M, D)
    c = conviction[:, None]
    stance_p = c * stance_u + (1 - c) * dominant
    stance_p += rng.normal(0, noise_sigma, size=stance_p.shape)
    return stance_p


def generate_posts(
    authors: np.ndarray,
    pop: Population,
    expression: ExpressionMap,
    s_t: np.ndarray,
    sigma_t: np.ndarray,
    eta: float,
    rng: np.random.Generator,
    noise_sigma_stance: float = 0.1,
    noise_sigma_d: float = 0.15,
) -> PostBatch:
    names = pop.trait_names
    topic_cols = [i for i, n in enumerate(names) if n.startswith("topic_affinity_")]
    stance_cols = [i for i, n in enumerate(names) if n.startswith("stance_")]
    conviction_col = names.index("conviction")

    topic_affinity = pop.X_used[authors][:, topic_cols]
    stance_u = pop.X_used[authors][:, stance_cols]
    conviction = pop.X_used[authors, conviction_col]

    topic_p = sample_topics(rng, topic_affinity, s_t, eta)
    stance_p = sample_stance(rng, conviction, stance_u, topic_p, sigma_t, noise_sigma_stance)

    d = expression.generate(pop.X_stored[authors], topic_p, s_t, rng, noise_sigma_d)

    return PostBatch(
        author=authors,
        topic=topic_p,
        stance=stance_p,
        arousal=d["arousal"],
        valence=d["valence"],
        provocativeness=d["provocativeness"],
        novelty=d["novelty"],
        specificity=d["specificity"],
        quality=d["quality"],
        length=d["length"],
    )
