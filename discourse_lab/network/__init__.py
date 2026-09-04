"""Graph generation (spec §2.2, dev §6 step 4): swappable generators sharing
one interface, `(cfg, pop, rng) -> csr_matrix`, registered by name so a
config's `graph.generator` string picks the theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from discourse_lab.config import Config
from discourse_lab.io.artifacts import artifact_paths, get_or_build
from discourse_lab.population import Population
from discourse_lab.registry import get
from discourse_lab.network.reciprocity import add_reciprocity

# import for registration side effects
from discourse_lab.network import barabasi, configuration, latent_space, sbm  # noqa: F401,E402


@dataclass
class Graph:
    csr: sparse.csr_matrix  # row u -> followees of u (fast "who does u follow")
    csc: sparse.csc_matrix  # col v -> followers of v (fast "who follows v")

    @property
    def n(self) -> int:
        return self.csr.shape[0]


def generate_graph(cfg: Config, pop: Population, rng: np.random.Generator) -> Graph:
    builder = get("graph_generator", cfg.graph.generator)
    G = builder(cfg, pop, rng)
    G = add_reciprocity(G, cfg.graph.reciprocity, rng)
    return Graph(csr=G.tocsr(), csc=G.tocsc())


def _save(path: Path, graph: Graph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, graph.csr)


def _load(path: Path) -> Graph:
    csr = sparse.load_npz(path)
    return Graph(csr=csr, csc=csr.tocsc())


def cached_graph(cfg: Config, seed: int, pop: Population, rng: np.random.Generator) -> Graph:
    root = artifact_paths(cfg)["graph"]
    path = root / f"{seed}.npz"
    return get_or_build(path, lambda: generate_graph(cfg, pop, rng), _save, _load)
