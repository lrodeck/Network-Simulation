"""The intervention study: which platform design choices move the normative
outcomes, measured against matched nulls.

Where Experiment 1 sweeps *theories* of engagement (which kernel), this sweeps
*design levers* — the things a platform actually controls — and reads
`outcomes.normative_outcomes` rather than the per-tick metrics. That is the
comparative deliverable spec §5.1's stylized facts are only a validity gate for.

Each cell is paired with a `kernel="null"` config that matches it in every
other respect, including the lever. So the reported effect is the lever's
contribution *given* that the engagement kernel is doing nothing — spec §5.3's
protocol, which exists because "heavy-tailed activity alone manufactures most
of what naively looks like emergent structure".

Levers, and what is known about each from measurement (n_users=800, 25 ticks,
exposure_sample_rate=0.10, 4 seeds, cross-camp exposure share, chance = 0.50):

    ranker           chronological 0.362 -> affinity 0.288. The dominant lever.
    long_tie_fraction 0.1 -> 0.4 raises it to 0.315. Network structure, not feed.
    inject_k         barely moves the aggregate (0.288 -> 0.292 at k=5), but the
                     injected items themselves are far more cross-cutting
                     (0.396). The mechanism works per item and is drowned out by
                     follower fanout at any dosage a platform would ship — which
                     is itself the finding, and why `algorithmic_share` is
                     reported separately.
    attention_budget 30 -> 100 moves it 0.288 -> 0.296. Weak.
    homophily_beta   0.35 -> 1.5 moves it 0.288 -> 0.280. Effectively inert, as
                     it is for clustering; the knob named "homophily strength"
                     is not the one that controls homophily.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import polars as pl

import discourse_lab.exposure  # noqa: F401  (registers the ranker table)
from discourse_lab.analysis import MIN_SEEDS, get_param, set_param
from discourse_lab.config import Config
from discourse_lab.outcomes import normative_outcomes, outcome_names
from discourse_lab.population import cached_population
from discourse_lab.runner import cached_run, load_run, phase_rngs
from discourse_lab.semantics import lexicon_for

# Every outcome needs the raw tables; a run persisted without them silently
# drops constructs, which in a comparative study means an empty column rather
# than a wrong number — but an empty column is still a wasted sweep.
PERSIST = ("posts", "engagements", "exposures", "traits")

DEFAULT_LEVERS: dict[str, tuple] = {
    "dynamics.ranker": ("chronological", "affinity", "engagement_optimized", "popularity"),
    "dynamics.inject_k": (0, 5, 20),
    "dynamics.attention_budget": (15.0, 30.0, 60.0),
    "graph.long_tie_fraction": (0.05, 0.1, 0.4),
}


@dataclass(frozen=True)
class InterventionCell:
    lever: str
    value: object
    cfg: Config
    null_cfg: Config


def build_interventions(
    base_cfg: Config, levers: Mapping[str, Sequence] = DEFAULT_LEVERS
) -> list[InterventionCell]:
    """One cell per (lever, value), each with a matched `kernel="null"` twin.

    One-at-a-time, not a full factorial: with four levers this is 13 cells
    rather than 108, and spec §5.4 asks for one-at-a-time sensitivity anyway.
    Interactions are a follow-up, not the first result.
    """
    cells = []
    for lever, values in levers.items():
        for value in values:
            cfg = set_param(base_cfg, lever, value)
            cfg = dataclasses.replace(cfg, label=f"intv-{lever}={value}")
            null_cfg = dataclasses.replace(
                set_param(cfg, "dynamics.kernel", "null"),
                label=f"intv-null-{lever}={value}",
            )
            cells.append(InterventionCell(lever=lever, value=value, cfg=cfg, null_cfg=null_cfg))
    return cells


def _outcomes_for(cfg: Config, seed: int) -> dict[str, float]:
    cached_run(cfg, seed, persist=PERSIST)
    handle = load_run(cfg, seed)
    pop = cached_population(cfg, seed, phase_rngs(seed)["population"])
    return normative_outcomes(handle, pop=pop, lex=lexicon_for(cfg))


def run_interventions(
    cells: Sequence[InterventionCell], seeds: Sequence[int], warn_on_few_seeds: bool = True
) -> pl.DataFrame:
    """Tidy long-form: one row per (cell, seed, outcome), with the model value,
    its matched-null value, and the difference.

    Two contrasts, and confusing them is easy:

    `kernel_delta` = model - matched null. The null holds the LEVER fixed and
    sets `kernel="null"`, so this isolates what the engagement kernel
    contributed *at this lever setting*. It is spec §5.3's protocol and it is a
    diagnostic here, not the headline: measured, the ranker moves cross-camp
    exposure from 0.387 to 0.309 while its kernel_delta is -0.004 either way,
    because the feed effect is present in both arms.

    The headline is the contrast BETWEEN lever values, which
    `summarize_interventions` computes against each lever's reference value.
    """
    seeds = list(seeds)
    if warn_on_few_seeds and len(seeds) < MIN_SEEDS:
        import warnings

        warnings.warn(
            f"{len(seeds)} seeds; spec §4.4 asks for at least {MIN_SEEDS}. Cascade "
            "dynamics have enormous run-to-run variance and a single run per "
            "condition tells you essentially nothing.",
            stacklevel=2,
        )

    rows: list[dict] = []
    for cell in cells:
        for seed in seeds:
            model = _outcomes_for(cell.cfg, seed)
            null = _outcomes_for(cell.null_cfg, seed)
            for name, value in model.items():
                if name.endswith(".n") or name.endswith(".delta"):
                    continue    # bookkeeping, not an outcome
                rows.append({
                    "lever": cell.lever,
                    "value": str(cell.value),
                    "seed": seed,
                    "outcome": name,
                    "model": float(value),
                    "null": float(null.get(name, float("nan"))),
                    "kernel_delta": float(value) - float(null.get(name, float("nan"))),
                })
    return pl.DataFrame(rows)


def summarize_interventions(
    rows: pl.DataFrame, reference: Mapping[str, object] | None = None
) -> pl.DataFrame:
    """Mean and spread across seeds, with the effect of each lever value
    measured against that lever's reference setting.

    `lever_effect` is the headline: outcome at this value minus outcome at the
    reference value, which is what "changing this design choice does X" means.
    `kernel_delta` is kept alongside as spec §5.3's diagnostic — whether the
    effect is mediated by the engagement kernel — and is NOT the same thing.
    Measured, the ranker moves cross-camp exposure by 0.078 between values
    while its kernel_delta is -0.004, because the matched null holds the lever
    fixed and so carries the same feed effect.

    `resolves` says whether `lever_effect` clears twice its own standard error
    across seeds. Reported, never filtered on: "this design choice does not move
    this outcome at a sample size a study can afford" is a result a normative
    paper needs to be able to state.
    """
    reference = dict(reference or {})
    grouped = (
        rows.group_by(["lever", "value", "outcome"])
        .agg(
            pl.col("model").mean().alias("model_mean"),
            pl.col("model").std().alias("model_sd"),
            pl.col("null").mean().alias("null_mean"),
            pl.col("kernel_delta").mean().alias("kernel_delta"),
            pl.len().alias("n_seeds"),
        )
        .sort(["outcome", "lever", "value"])
    )

    # the reference value per lever: caller's choice, else the first value seen
    first_seen = {}
    for lever, value in rows.select(["lever", "value"]).unique(maintain_order=True).iter_rows():
        first_seen.setdefault(lever, value)
    refs = {lever: str(reference.get(lever, default)) for lever, default in first_seen.items()}

    baseline = (
        grouped.with_columns(
            pl.col("lever").replace_strict(refs, default=None).alias("_ref")
        )
        .filter(pl.col("value") == pl.col("_ref"))
        .select(["lever", "outcome", pl.col("model_mean").alias("_base"),
                 pl.col("model_sd").alias("_base_sd")])
    )

    return (
        grouped.join(baseline, on=["lever", "outcome"], how="left")
        .with_columns(
            (pl.col("model_mean") - pl.col("_base")).alias("lever_effect"),
            pl.col("lever").replace_strict(refs, default=None).alias("reference"),
        )
        .with_columns(
            (
                pl.col("lever_effect").abs()
                > 2 * (pl.col("model_sd").pow(2) + pl.col("_base_sd").pow(2)).sqrt()
                / pl.col("n_seeds").sqrt()
            ).alias("resolves")
        )
        .drop(["_base", "_base_sd"])
        .sort(["outcome", "lever", "value"])
    )
