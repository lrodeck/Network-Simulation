"""Archetype mixture (spec §2.1): declarative offsets from the population mean.

Membership is retained as a label for analysis but is never read by the
dynamics — it only shapes the initial draw. Communities must emerge from
trait geometry, not a group ID the model could cheat with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# offsets in latent (z-space) units, applied before the copula transform.
DEFAULT_ARCHETYPES: tuple["Archetype", ...] = ()


@dataclass(frozen=True)
class Archetype:
    name: str
    w: float
    offsets: dict[str, float] = field(default_factory=dict)


DEFAULT_ARCHETYPES = (
    Archetype("lurker", w=0.55, offsets={"activity": -1.5, "reply_prop": -1.0}),
    Archetype("poster", w=0.25, offsets={"activity": 0.8}),
    Archetype("firebrand", w=0.08, offsets={"contrarianism": 1.5}),
    Archetype("institution", w=0.02, offsets={"prominence": 2.5, "formality": 1.5}),
    Archetype("newcomer", w=0.10, offsets={"plasticity": 1.5, "conviction": -1.0}),
)


def resolve_archetypes(
    weighted_names: tuple[tuple[str, float], ...],
    offsets: tuple[tuple[str, str, float], ...],
) -> tuple[Archetype, ...]:
    """Build the archetype library from config overrides, or the library
    default when the config leaves both empty.
    """
    if not weighted_names and not offsets:
        return DEFAULT_ARCHETYPES

    by_name = {a.name: dict(a.offsets) for a in DEFAULT_ARCHETYPES}
    weights = dict(weighted_names) if weighted_names else {a.name: a.w for a in DEFAULT_ARCHETYPES}
    for name, trait, value in offsets:
        by_name.setdefault(name, {})[trait] = value

    return tuple(Archetype(name=n, w=w, offsets=by_name.get(n, {})) for n, w in weights.items())


def archetype_component_means(
    archetypes: tuple[Archetype, ...], trait_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Weights and (C x n) mean matrix, offsets applied only to traits that
    exist in this population's trait table — an archetype's other offsets
    (e.g. against a post-level trait) are silently skipped.
    """
    idx = {name: i for i, name in enumerate(trait_names)}
    weights = np.array([a.w for a in archetypes], dtype=float)
    weights = weights / weights.sum()

    means = np.zeros((len(archetypes), len(trait_names)))
    for c, a in enumerate(archetypes):
        for trait, value in a.offsets.items():
            i = idx.get(trait)
            if i is not None:
                means[c, i] = value
    return weights, means
