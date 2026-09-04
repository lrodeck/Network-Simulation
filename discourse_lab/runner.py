"""Run loop skeleton: phase RNG discipline, run_iter/run/cached_run (dev §6 step 1, spec §3.4).

`run_iter` is the generator core; `run` collects it into a persisted, cached
run directory. Interactive callers (the run monitor) consume `run_iter`
directly to inspect or break mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from discourse_lab.config import Config
from discourse_lab.io.store import RunHandle, RunWriter
from discourse_lab.io.workspace import runs_dir

# Fixed order: spawning child generators from this list must stay stable across
# versions, or resuming an old seed produces different streams than it did
# originally. Append new phases at the end; never reorder or remove existing
# ones.
PHASES: tuple[str, ...] = (
    "population",
    "graph",
    "timing",
    "generation",
    "exposure",
    "reaction",
    "perception",
    "cascade",
    "drift",
)


def phase_rngs(seed: int) -> dict[str, np.random.Generator]:
    """One root seed, independent child generator per phase (spec §3.4).

    Phase-independent streams mean a change confined to one phase's draws
    (e.g. how many posts render in tick 3) cannot shift the stream any other
    phase consumes, so parameter sweeps are not confounded by RNG misalignment.
    """
    root = np.random.default_rng(seed)
    children = root.spawn(len(PHASES))
    return {name: np.random.default_rng(s) for name, s in zip(PHASES, children)}


@dataclass
class State:
    """One tick's worth of run state, as seen by run_iter consumers."""

    t: int
    cfg: Config
    seed: int
    rngs: dict[str, np.random.Generator]
    metrics: dict[str, float]


def run_dir(cfg: Config, seed: int) -> Path:
    return runs_dir() / cfg.hash() / str(seed)


def run_iter(cfg: Config, seed: int) -> Iterator[State]:
    """Generator core. Empty tick body — phases land in later build steps."""
    rngs = phase_rngs(seed)
    for t in range(cfg.dynamics.n_ticks):
        # phases hook in here, one per build step (dev §6); nothing yet runs.
        yield State(t=t, cfg=cfg, seed=seed, rngs=rngs, metrics={})


def run(cfg: Config, seed: int) -> Path:
    """Collect run_iter, streaming each tick to parquet. Returns the run dir."""
    path = run_dir(cfg, seed)
    writer = RunWriter(path, cfg, seed)
    try:
        for state in run_iter(cfg, seed):
            writer.write_tick(state.t, state.metrics)
    finally:
        writer.close()
    return path


def cached_run(cfg: Config, seed: int) -> Path:
    """Reuse a complete prior run for this exact (config, seed); else run it."""
    path = run_dir(cfg, seed)
    if (path / "COMPLETE").exists():
        return path
    return run(cfg, seed)


def load_run(cfg: Config, seed: int) -> RunHandle:
    return RunHandle(run_dir(cfg, seed))
