"""Run loop skeleton: phase RNG discipline, run_iter/run/cached_run (dev §6 step 1, spec §3.4).

`run_iter` is the generator core; `run` collects it into a persisted, cached
run directory. Interactive callers (the run monitor) consume `run_iter`
directly to inspect or break mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from discourse_lab.config import Config
from discourse_lab.dynamics.posts import PostBatch, concat_post_batches
from discourse_lab.dynamics.tick import TickEngine
from discourse_lab.io.store import (
    RUN_FORMAT,
    RunHandle,
    RunWriter,
    engagements_schema,
    exposures_schema,
    posts_schema,
    salient_events_schema,
    traits_schema,
)
from discourse_lab.io.workspace import runs_dir
from discourse_lab.network import cached_graph
from discourse_lab.population import cached_population

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
    # Raw record for this tick, for persistence or for interactive callers who
    # want it without a writer. `retired_posts` carries final engagement
    # counts; on the last tick it also carries everything still active, so no
    # post is lost. Both are per-tick, never cumulative.
    retired_posts: "PostBatch | None" = None
    engagement_events: dict[str, np.ndarray] | None = None
    exposure_sample: dict[str, np.ndarray] | None = None
    # spec §3.5: X snapshots every `snapshot_every` ticks. None on other ticks.
    traits_snapshot: np.ndarray | None = None
    # spec §3.1 step 6: THIS TICK's salient events, queued for the offline
    # channel-3 pass and never executed inside the tick. Replaced every tick, like
    # the other raw records — a consumer that wants the whole run accumulates.
    salient_events: list = field(default_factory=list)


def run_dir(cfg: Config, seed: int) -> Path:
    return runs_dir() / cfg.hash() / str(seed)


def run_iter(cfg: Config, seed: int) -> Iterator[State]:
    """Generator core: population/graph are cached artifacts, then the tick
    engine (discourse_lab.dynamics.tick.TickEngine) runs timing, generation,
    exposure, reaction, cascades, perception, and the discourse-state update
    each tick. Drift (step 11) is not wired in yet, so traits are static.
    """
    rngs = phase_rngs(seed)
    pop = cached_population(cfg, seed, rngs["population"])
    graph = cached_graph(cfg, seed, pop, rngs["graph"])
    engine = TickEngine(cfg=cfg, pop=pop, graph=graph, rngs=rngs)

    last_tick = cfg.dynamics.n_ticks - 1
    for t in range(cfg.dynamics.n_ticks):
        metrics = engine.step(t)
        retired = engine.retired_posts
        if t == last_tick and engine.active_posts is not None and len(engine.active_posts) > 0:
            # final flush: everything still in the pool retires with the run,
            # so `run()` stays a pure consumer and no post goes unrecorded
            retired = (
                concat_post_batches([retired, engine.active_posts])
                if retired is not None and len(retired) > 0
                else engine.active_posts
            )
        yield State(
            t=t, cfg=cfg, seed=seed, rngs=rngs, metrics=metrics,
            retired_posts=retired, engagement_events=engine.engagement_events,
            exposure_sample=engine.exposure_sample,
            traits_snapshot=(
                engine.pop.X_stored.copy()
                if cfg.dynamics.snapshot_every > 0 and t % cfg.dynamics.snapshot_every == 0
                else None
            ),
            salient_events=engine.salient_events,
        )


def run(cfg: Config, seed: int, persist: Sequence[str] = ()) -> Path:
    """Collect run_iter, streaming each tick to parquet. Returns the run dir.

    `persist` opts into the optional raw tables: `("posts", "engagements")`.
    It is an argument rather than a config field on purpose — persistence is
    an output concern, and putting it in `DynamicsConfig` would change
    `cfg.hash()` and invalidate every cached artifact.
    """
    persist = tuple(persist)
    unknown = set(persist) - {"posts", "engagements", "exposures", "traits", "salient_events"}
    if unknown:
        raise ValueError(
            f"unknown persist target(s): {sorted(unknown)}; "
            "expected any of 'posts', 'engagements', 'exposures', 'traits', 'salient_events'"
        )

    path = run_dir(cfg, seed)
    writer = RunWriter(path, cfg, seed)
    try:
        # create the files up front so an empty run is distinguishable from
        # persistence having been switched off
        if "posts" in persist:
            writer.ensure_empty("posts", posts_schema(cfg.stance_dims()))
        if "engagements" in persist:
            writer.ensure_empty("engagements", engagements_schema())
        if "exposures" in persist:
            writer.ensure_empty("exposures", exposures_schema())
        if "salient_events" in persist:
            writer.ensure_empty("salient_events", salient_events_schema())

        for state in run_iter(cfg, seed):
            writer.write_tick(state.t, state.metrics)
            if "posts" in persist:
                writer.write_posts(state.retired_posts)
            if "engagements" in persist:
                writer.write_engagements(state.engagement_events)
            if "exposures" in persist:
                writer.write_exposure_sample(state.exposure_sample)
            if "salient_events" in persist:
                writer.write_salient_events(state.t, state.salient_events)
            if "traits" in persist and state.traits_snapshot is not None:
                # created on first snapshot rather than up front: the trait
                # count is only known once the population is sampled
                writer.write_traits(state.t, state.traits_snapshot)
    finally:
        writer.close()
    return path


def cached_run(cfg: Config, seed: int, persist: Sequence[str] = ()) -> Path:
    """Reuse a complete prior run for this exact (config, seed); else run it.

    A directory is only reused if it was written at the current `RUN_FORMAT`
    *and* already has whatever `persist` asks for — otherwise the run is
    redone. Population and graph artifacts are keyed on their own sub-hashes
    and survive, so only the dynamics recompute.
    """
    path = run_dir(cfg, seed)
    if (path / "COMPLETE").exists():
        handle = RunHandle(path)
        fresh_enough = handle.format >= RUN_FORMAT
        has_what_was_asked = all(
            getattr(handle, f"has_{target}") for target in persist
        )
        if fresh_enough and has_what_was_asked:
            return path
    return run(cfg, seed, persist=persist)


def load_run(cfg: Config, seed: int) -> RunHandle:
    return RunHandle(run_dir(cfg, seed))
