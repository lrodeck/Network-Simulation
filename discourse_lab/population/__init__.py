"""Population generation (spec §2.1, dev §6 step 2): archetype mixture over a
correlated Gaussian latent, transformed to target marginals by a Gaussian
copula, stored in the unconstrained space drift will later operate on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import warnings

import numpy as np
from scipy import stats

from discourse_lab.config import Config
from discourse_lab.io.artifacts import artifact_paths, get_or_build
from discourse_lab.population.archetypes import archetype_component_means, resolve_archetypes
from discourse_lab.population.copula import (
    mixture_moments,
    nearest_psd_correlation,
    sample_latent,
    sparse_pairs_to_matrix,
)
from discourse_lab.population.links import to_stored
from discourse_lab.population.traits import TraitSpec, trait_table


@dataclass
class Population:
    trait_names: list[str]
    X_used: np.ndarray     # N x n, constrained space
    X_stored: np.ndarray   # N x n, unconstrained space drift operates on
    links: list[str]       # per-column link kind ("identity"|"logit"|"log"|"tanh"), so
                            # drift (dynamics/drift.py) can recompute X_used from X_stored
    archetype_labels: np.ndarray
    archetype_names: list[str]


def _warn_on_composed_correlation(pairs, archetypes) -> None:
    """Warn when a requested correlation touches traits an archetype also moves.

    The two mechanisms compose additively and neither knows about the other, so
    `correlation_pairs` reads as "set this correlation to X" while behaving as
    "add X on top of whatever the mixture already produces". Measured at
    N=20000 on the shipped defaults:

        archetypes off, no pairs        activity x reply_prop = -0.001
        archetypes off, asked for 0.30                        = +0.299
        archetypes on,  no pairs                              = +0.303
        archetypes on,  asked for 0.30                        = +0.493

    Asking for 0.30 and getting 0.49 is the kind of thing that survives into a
    paper as "we set trait correlation to 0.3", so it is worth a warning rather
    than a docstring nobody reads at the call site.
    """
    if not pairs:
        return

    moved: dict[str, list[str]] = {}
    for archetype in archetypes:
        for trait in archetype.offsets:
            moved.setdefault(trait, []).append(archetype.name)

    for a, b, rho in pairs:
        if a in moved and b in moved:
            shared = sorted(set(moved[a]) & set(moved[b]))
            if shared:
                warnings.warn(
                    f"correlation_pairs asks for {a} x {b} = {rho:+.2f}, but the "
                    f"archetype(s) {shared} shift both traits and so already correlate "
                    "them. The two mechanisms add: the realised correlation will be "
                    "larger than requested. Set archetype_offsets or correlation_pairs, "
                    "not both, for a pair you want to control exactly.",
                    stacklevel=3,
                )


def sample_population(cfg: Config, rng: np.random.Generator) -> Population:
    specs: list[TraitSpec] = trait_table(cfg)
    names = [s.name for s in specs]
    n_traits = len(names)
    n_users = cfg.population.n_users

    corr = sparse_pairs_to_matrix(names, cfg.population.correlation_pairs)
    corr = nearest_psd_correlation(corr)

    archetypes = resolve_archetypes(cfg.population.archetype_weights, cfg.population.archetype_offsets)
    _warn_on_composed_correlation(cfg.population.correlation_pairs, archetypes)
    weights, component_means = archetype_component_means(archetypes, names)

    z, labels = sample_latent(rng, weights, component_means, corr, n_users)

    mu_bar, sigma_bar = mixture_moments(weights, component_means, np.diag(corr))
    w = stats.norm.cdf((z - mu_bar) / sigma_bar)

    X_used = np.empty_like(w)
    X_stored = np.empty_like(w)
    for i, spec in enumerate(specs):
        X_used[:, i] = spec.marginal.icdf(w[:, i])
        X_stored[:, i] = to_stored(X_used[:, i], spec.link)

    return Population(
        trait_names=names,
        X_used=X_used,
        X_stored=X_stored,
        links=[spec.link for spec in specs],
        archetype_labels=labels,
        archetype_names=[a.name for a in archetypes],
    )


def _save(path: Path, pop: Population) -> None:
    np.savez(
        path,
        trait_names=np.array(pop.trait_names),
        X_used=pop.X_used,
        X_stored=pop.X_stored,
        links=np.array(pop.links),
        archetype_labels=pop.archetype_labels,
        archetype_names=np.array(pop.archetype_names),
    )


def _load(path: Path) -> Population:
    data = np.load(path, allow_pickle=False)
    return Population(
        trait_names=list(data["trait_names"]),
        X_used=data["X_used"],
        X_stored=data["X_stored"],
        links=list(data["links"]),
        archetype_labels=data["archetype_labels"],
        archetype_names=list(data["archetype_names"]),
    )


def cached_population(cfg: Config, seed: int, rng: np.random.Generator) -> Population:
    """Content-addressed cache keyed by the population/scenario sub-hash and
    seed (dev §8.2): reused unchanged when dynamics-only config fields change.
    """
    root = artifact_paths(cfg)["population"]
    path = root / f"{seed}.npz"
    return get_or_build(path, lambda: sample_population(cfg, rng), _save, _load)
