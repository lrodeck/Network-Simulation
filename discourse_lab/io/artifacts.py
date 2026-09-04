"""Content-addressed artifact cache (dev notes §8.2).

Population and graph are cached as artifacts keyed by their structural
sub-hash, never shared as mutable objects between runs:

    dlab/artifacts/pop/{pop_hash}/{seed}.npz
    dlab/artifacts/graph/{pop_hash}-{graph_hash}/{seed}.npz

Because the key is derived from exactly the config that determines the
artifact, the cache is correct by construction: changing a population field
invalidates the population artifact (and the graph that depends on it), while
dynamics edits invalidate nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from discourse_lab.config import Config, structural_hash
from discourse_lab.io.workspace import artifacts_dir


def population_key(cfg: Config) -> str:
    return structural_hash((cfg.population, cfg.scenario))


def graph_key(cfg: Config) -> str:
    return f"{population_key(cfg)}-{structural_hash(cfg.graph)}"


def artifact_paths(cfg: Config) -> dict[str, Path]:
    root = artifacts_dir()
    return {
        "population": root / "pop" / population_key(cfg),
        "graph": root / "graph" / graph_key(cfg),
    }


def get_or_build(
    path: Path,
    builder: Callable[[], Any],
    saver: Callable[[Path, Any], None],
    loader: Callable[[Path], Any],
) -> Any:
    if path.exists():
        return loader(path)
    obj = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    saver(path, obj)
    return obj
