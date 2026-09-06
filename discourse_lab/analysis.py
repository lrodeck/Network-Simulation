"""Grid sweeps and one-at-a-time sensitivity (spec §4.4, §5.4).

`sweep.sweep` runs (config, seed) pairs and returns run directories, which is
the right primitive for parallel execution but not the API §4.4 describes:

    sweep(base: Config, grid: dict[str, list], seeds: list[int]) -> pl.DataFrame
        \"\"\"Returns tidy long-form: one row per (config, seed, metric).\"\"\"

This module is that layer: it expands a grid of dotted parameter paths over a
base config, runs the cells, and returns the tidy frame the notebook and the
figures consume.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from discourse_lab.config import Config
from discourse_lab.runner import cached_run, load_run

# spec §4.4: "Minimum 10 seeds, and report distributions rather than points.
# This is the single most common way simulation studies of this kind go wrong."
MIN_SEEDS = 10

# spec §5.4's named one-at-a-time parameters. `hawkes_alpha_beta` is
# `hawkes_ratio` here (it is literally alpha/beta); `ou_k` is a tuple of
# per-block overrides rather than a scalar, so it is swept via `drift_lr`,
# the scalar that actually sets drift step size.
SENSITIVITY_PARAMS = (
    "dynamics.attention_budget",
    "dynamics.inject_k",
    "dynamics.hawkes_ratio",
    "dynamics.drift_lr",
    "graph.mean_degree",
)


def set_param(cfg: Config, path: str, value: Any) -> Config:
    """`set_param(cfg, "dynamics.inject_k", 5)` -> a new frozen Config."""
    head, _, tail = path.partition(".")
    if not tail:
        return dataclasses.replace(cfg, **{head: value})
    section = getattr(cfg, head)
    return dataclasses.replace(cfg, **{head: dataclasses.replace(section, **{tail: value})})


def get_param(cfg: Config, path: str) -> Any:
    obj: Any = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def expand_grid(base: Config, grid: Mapping[str, Sequence[Any]]) -> list[tuple[dict, Config]]:
    """Cartesian product over dotted parameter paths. Returns (cell, cfg)."""
    if not grid:
        return [({}, base)]
    keys = list(grid)
    cells = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = base
        for k, v in zip(keys, combo):
            cfg = set_param(cfg, k, v)
        cells.append((dict(zip(keys, combo)), cfg))
    return cells


def sweep(
    base: Config,
    grid: Mapping[str, Sequence[Any]],
    seeds: Sequence[int],
    metrics: Sequence[str] | None = None,
    warn_on_few_seeds: bool = True,
) -> pl.DataFrame:
    """Run the grid and return tidy long-form: one row per (cell, seed, metric).

    Each metric is the mean over the run's ticks, which is what the §5.2/§5.3
    comparisons operate on. Keeping it long-form rather than wide means adding
    a metric never changes the schema.
    """
    seeds = list(seeds)
    if warn_on_few_seeds and len(seeds) < MIN_SEEDS:
        import warnings

        warnings.warn(
            f"sweeping over {len(seeds)} seeds; spec §4.4 asks for at least {MIN_SEEDS}. "
            "Cascade dynamics have enormous run-to-run variance and a single run per "
            "condition tells you essentially nothing.",
            stacklevel=2,
        )

    rows: list[dict] = []
    for cell, cfg in expand_grid(base, grid):
        for seed in seeds:
            cached_run(cfg, seed)
            frame = load_run(cfg, seed).metrics()
            names = metrics or [c for c in frame.columns if c != "t"]
            for name in names:
                if name not in frame.columns:
                    continue
                values = frame[name].to_numpy().astype(float)
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        **{k: str(v) for k, v in cell.items()},
                        "seed": seed,
                        "metric": name,
                        "value": float(finite.mean()) if len(finite) else float("nan"),
                    }
                )
    return pl.DataFrame(rows)


def sensitivity(
    base: Config,
    seeds: Sequence[int],
    params: Sequence[str] = SENSITIVITY_PARAMS,
    factors: Sequence[float] = (0.5, 1.0, 2.0),
    metrics: Sequence[str] | None = None,
) -> pl.DataFrame:
    """One-at-a-time sensitivity (spec §5.4).

        "Any conclusion that inverts within a plausible range of one of these
        is a conclusion about that parameter, not about the mechanism under
        study."

    Each parameter is scaled by `factors` around its base value while every
    other parameter is held fixed. Returns tidy long-form with the parameter,
    its value, the multiplier, seed, metric and value, plus `inverts` — whether
    the metric changed sign across the swept range, which is the specific
    failure §5.4 exists to catch.
    """
    frames = []
    for path in params:
        base_value = get_param(base, path)
        values = []
        for f in factors:
            scaled = base_value * f
            values.append(type(base_value)(scaled) if isinstance(base_value, int) else scaled)
        values = sorted(set(values))
        frame = sweep(base, {path: values}, seeds, metrics=metrics, warn_on_few_seeds=False)
        frames.append(frame.with_columns(pl.lit(path).alias("param")).rename({path: "param_value"}))

    out = pl.concat(frames, how="diagonal")
    inverts = (
        out.group_by(["param", "metric", "param_value"])
        .agg(pl.col("value").mean().alias("cell_mean"))
        .group_by(["param", "metric"])
        .agg(
            ((pl.col("cell_mean").min() < 0) & (pl.col("cell_mean").max() > 0)).alias("inverts")
        )
    )
    return out.join(inverts, on=["param", "metric"], how="left")
