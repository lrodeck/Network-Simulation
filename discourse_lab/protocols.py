"""Structural interfaces for the swappable components (spec §4.2).

    "Every swappable component is a Protocol. This is what makes it a toolbox
    rather than one simulation."

These are `typing.Protocol`s, so they are structural: nothing has to inherit
from them and no runtime dispatch changes. They exist so that a component
written elsewhere can be checked against the contract the loop actually calls,
by a type checker or by `isinstance` (each is `runtime_checkable`), instead of
the contract living only in the call site.

Registration by name is the other half of §4.2 and already exists in
`discourse_lab.registry`: a config holds `kernel: str = "outrage"`, so a run
is fully described by JSON and sweeps over theories are sweeps over strings.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy import sparse


@runtime_checkable
class GraphGenerator(Protocol):
    """`network/*.py`, registered under "graph_generator"."""

    def __call__(self, cfg, pop, rng: np.random.Generator) -> sparse.csr_matrix: ...


@runtime_checkable
class Ranker(Protocol):
    """`exposure/rankers.py`, registered under "ranker". Returns one score per
    candidate pair; `rank_candidates` sorts and the attention budget cuts."""

    def __call__(self, pairs, posts, pop, rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class FeatureMap(Protocol):
    """The theory of engagement (spec §2.6): `phi` plus the theta that weights
    it. `exposure/kernel.py` supplies `compute_features` and KERNEL_THETAS;
    swapping a theory means swapping this pair and nothing else."""

    def __call__(self, exposures, posts, pop, is_follower: np.ndarray, t_current: int) -> dict: ...


@runtime_checkable
class DriftModel(Protocol):
    """`dynamics/drift.py`. Mutates the population in place rather than
    returning dX, because X is a single shared matrix (spec §1.1) and copying
    it every tick is the thing struct-of-arrays exists to avoid."""

    def __call__(self, cfg, pop, expr, state, rng: np.random.Generator, t: int,
                 posts, engagement_delta, exposures, actions) -> None: ...
