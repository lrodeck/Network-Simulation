"""Experiment 1 (dev §6 step 10): sweep engagement kernels and rankers with
multiple seeds, every effect measured against its matched `kernel="null"`
run — same population, same graph, same activity (spec §5.3's mandatory
protocol, made concrete here rather than left as a manual step).

Cells share a `label` prefix and everything except `kernel`/`ranker`, so
`cached_population`/`cached_graph` are reused across a whole cell's seeds
and its null counterpart (dev §8.2) — only the dynamics differ.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import polars as pl

import discourse_lab.exposure  # noqa: F401  (registers rankers and kernel_theta tables)
from discourse_lab.config import Config
from discourse_lab.exposure.kernel import kernel_names
from discourse_lab.metrics import null_comparison
from discourse_lab.registry import names
from discourse_lab.runner import cached_run, load_run

DEFAULT_KERNELS = tuple(k for k in kernel_names() if k != "null")
DEFAULT_RANKERS = tuple(names("ranker"))

TRACKED_METRICS = ("attention_gini", "bubble_index", "r_eff", "agreement", "salience")


@dataclass(frozen=True)
class Experiment1Cell:
    kernel: str
    ranker: str
    cfg: Config
    null_cfg: Config


def build_experiment1(
    base_cfg: Config,
    kernels: Sequence[str] = DEFAULT_KERNELS,
    rankers: Sequence[str] = DEFAULT_RANKERS,
) -> list[Experiment1Cell]:
    """One cell per (kernel, ranker) pair, each paired with a same-ranker
    `kernel="null"` config that otherwise matches exactly.
    """
    cells = []
    for kernel in kernels:
        for ranker in rankers:
            cfg = dataclasses.replace(
                base_cfg,
                dynamics=dataclasses.replace(base_cfg.dynamics, kernel=kernel, ranker=ranker),
                label=f"exp1-{kernel}-{ranker}",
            )
            null_cfg = dataclasses.replace(
                base_cfg,
                dynamics=dataclasses.replace(base_cfg.dynamics, kernel="null", ranker=ranker),
                label=f"exp1-null-{ranker}",
            )
            cells.append(Experiment1Cell(kernel=kernel, ranker=ranker, cfg=cfg, null_cfg=null_cfg))
    return cells


def run_experiment1(cells: list[Experiment1Cell], seeds: Sequence[int]) -> list[dict]:
    """Run every cell across every seed (a flat, seed-major, resumable sweep
    by construction: `cached_run` skips anything already on disk). Returns
    one row per (cell, seed) with each tracked metric's mean over the run and
    its null-comparison delta.
    """
    rows: list[dict] = []
    for cell in cells:
        for seed in seeds:
            cached_run(cell.cfg, seed)
            cached_run(cell.null_cfg, seed)
            metrics_effect = load_run(cell.cfg, seed).metrics()
            metrics_null = load_run(cell.null_cfg, seed).metrics()

            row = {"kernel": cell.kernel, "ranker": cell.ranker, "seed": seed}
            for col in TRACKED_METRICS:
                effect_val = float(metrics_effect[col].drop_nulls().mean() or float("nan"))
                null_val = float(metrics_null[col].drop_nulls().mean() or float("nan"))
                row[col] = effect_val
                row[f"{col}_null"] = null_val
                row[f"{col}_effect"] = null_comparison(effect_val, null_val)
            rows.append(row)
    return rows


def summarize_experiment1(rows: list[dict]) -> pl.DataFrame:
    """Distributions, not points (NOTES in TODO.txt): mean and std of each
    metric's null-comparison effect across seeds, grouped by (kernel, ranker).
    """
    df = pl.DataFrame(rows)
    effect_cols = [f"{c}_effect" for c in TRACKED_METRICS]
    agg = [pl.col(c).mean().alias(f"{c}_mean") for c in effect_cols] + [
        pl.col(c).std().alias(f"{c}_std") for c in effect_cols
    ]
    return df.group_by(["kernel", "ranker"], maintain_order=True).agg(agg)
