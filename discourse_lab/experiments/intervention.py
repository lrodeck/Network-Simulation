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
    attention_budget was swept at 15/30/60 and moved nothing on any of the 13
                     outcome columns over 10 seeds. That was a dead-zone sweep,
                     not an inert lever: visibility decays as exp(-r/tau_pos)
                     with tau_pos = 6, so the budget cap only binds on 10% of
                     surviving items at b=15, 1.2% at b=30 and 0.0% at b=60 --
                     30 and 60 are the same platform. Re-ranged to 30/10/3,
                     where it binds on 1%/25%/60%, and `tau_position` added
                     alongside it as the knob that actually rations attention.
    homophily_beta   0.35 -> 1.5 moves it 0.288 -> 0.280. Effectively inert, as
                     it is for clustering; the knob named "homophily strength"
                     is not the one that controls homophily.

**One lever at a time is a design choice, and it has a blind spot.** Measured,
`tau_position` (how far down the feed anyone reads) moves cross-camp exposure by
-0.004 under a chronological feed and by **+0.088** under an affinity feed, over
5 seeds at N=800 --- larger than every main effect in the 10-seed sweep except
the ranker itself. The reason is mechanical: rank order under `chronological` is
recency, which is independent of stance, so truncating the feed removes a random
slice; under `affinity` rank order *is* stance order, so truncating removes
precisely the disagreement. The main-effects sweep reports `tau_position` as
inert because it is evaluated at the base ranker, and averaging over a factor it
interacts with is how a real lever disappears.

So: the sweep below is a screen, not the study. A lever that reads flat here has
been shown flat *at the base configuration*, which is a weaker claim than "this
does not matter". Anything that survives the screen deserves a small factorial
against `dynamics.ranker` before it goes in a paper.
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
    # first value is the reference; both of these are the shipped default
    "dynamics.attention_budget": (30.0, 10.0, 3.0),
    "dynamics.tau_position": (6.0, 15.0, 2.0),
    "graph.long_tie_fraction": (0.05, 0.1, 0.4),
}


# The background factor a screened lever is most likely to interact with, and
# the one that has been measured to do so. See `across=` below.
RANKER_BACKGROUND: dict[str, tuple] = {
    "dynamics.ranker": ("chronological", "affinity"),
}


@dataclass(frozen=True)
class InterventionCell:
    lever: str
    value: object
    cfg: Config
    null_cfg: Config
    background: tuple[tuple[str, object], ...] = ()

    @property
    def background_label(self) -> str:
        """Canonical name for this cell's background, `""` when there is none.

        A string and not the tuple, because it becomes a group-by key in the
        tidy frame and a facet label in the figures.
        """
        return ", ".join(f"{k}={v}" for k, v in self.background)


def _background_grid(across: Mapping[str, Sequence] | None) -> list[tuple[tuple[str, object], ...]]:
    if not across:
        return [()]
    grid: list[tuple[tuple[str, object], ...]] = [()]
    for factor, levels in across.items():
        grid = [combo + ((factor, level),) for combo in grid for level in levels]
    return grid


def build_interventions(
    base_cfg: Config,
    levers: Mapping[str, Sequence] = DEFAULT_LEVERS,
    across: Mapping[str, Sequence] | None = None,
) -> list[InterventionCell]:
    """One cell per (background, lever, value), each with a matched
    `kernel="null"` twin.

    With `across=None` this is the one-at-a-time screen: five levers become 16
    cells rather than a 432-cell full factorial, and spec §5.4 asks for
    one-at-a-time sensitivity. **Read it as a screen.** A lever that comes out
    flat has been shown flat *at the base configuration*, which is a weaker
    claim than "this does not matter", and the difference is not academic —
    `tau_position` reads flat in the screen and moves cross-camp exposure by
    +0.088 under `affinity` against -0.004 under `chronological`.

    `across` re-runs every lever value inside each level of a background
    factor, so that interaction is visible instead of averaged away. Pass
    `RANKER_BACKGROUND` for the crossing that has been measured to matter:

        cells = build_interventions(cfg, across=RANKER_BACKGROUND)

    Cost is multiplicative and worth stating before you launch it: `across`
    with two levels doubles the cells, and each cell is a model run plus a null
    run per seed. The 16-cell screen at N=1500, 30 ticks, 10 seeds took ~180s;
    crossed against two rankers it is ~2x that.

    A lever that is *also* a background factor is not crossed with itself — its
    values already vary across the background, so it contributes one cell per
    value with no extra crossing, and `lever_effect` for it is the main effect.
    """
    cells = []
    for background in _background_grid(across):
        applied = base_cfg
        for factor, level in background:
            applied = set_param(applied, factor, level)
        crossed = {factor for factor, _ in background}
        prefix = "".join(f"{k}={v}|" for k, v in background)

        for lever, values in levers.items():
            if lever in crossed:
                # already varied by the background; crossing it with itself
                # would silently overwrite the background level
                continue
            for value in values:
                cfg = set_param(applied, lever, value)
                cfg = dataclasses.replace(cfg, label=f"intv-{prefix}{lever}={value}")
                null_cfg = dataclasses.replace(
                    set_param(cfg, "dynamics.kernel", "null"),
                    label=f"intv-null-{prefix}{lever}={value}",
                )
                cells.append(InterventionCell(lever=lever, value=value, cfg=cfg,
                                              null_cfg=null_cfg, background=background))

        for factor, levels in levers.items():
            if factor not in crossed:
                continue
            value = dict(background)[factor]
            if value not in levels:
                continue
            cfg = dataclasses.replace(applied, label=f"intv-{prefix}{factor}")
            null_cfg = dataclasses.replace(
                set_param(cfg, "dynamics.kernel", "null"),
                label=f"intv-null-{prefix}{factor}",
            )
            cells.append(InterventionCell(lever=factor, value=value, cfg=cfg,
                                          null_cfg=null_cfg, background=()))
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
                    "background": cell.background_label,
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
    if "background" not in rows.columns:
        rows = rows.with_columns(pl.lit("").alias("background"))

    grouped = (
        rows.group_by(["lever", "background", "value", "outcome"])
        .agg(
            pl.col("model").mean().alias("model_mean"),
            pl.col("model").std().alias("model_sd"),
            pl.col("null").mean().alias("null_mean"),
            pl.col("kernel_delta").mean().alias("kernel_delta"),
            pl.len().alias("n_seeds"),
        )
        .sort(["outcome", "lever", "background", "value"])
    )

    # the reference value per lever: caller's choice, else the first value seen.
    # Per lever and not per (lever, background): the reference is a property of
    # the dial, and a background-specific reference would make lever_effect
    # incomparable across backgrounds, which is the one comparison `across=`
    # exists to support.
    first_seen = {}
    for lever, value in rows.select(["lever", "value"]).unique(maintain_order=True).iter_rows():
        first_seen.setdefault(lever, value)
    refs = {lever: str(reference.get(lever, default)) for lever, default in first_seen.items()}

    baseline = (
        grouped.with_columns(
            pl.col("lever").replace_strict(refs, default=None).alias("_ref")
        )
        .filter(pl.col("value") == pl.col("_ref"))
        .select(["lever", "background", "outcome",
                 pl.col("model_mean").alias("_base"),
                 pl.col("model_sd").alias("_base_sd")])
    )

    return (
        grouped.join(baseline, on=["lever", "background", "outcome"], how="left")
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
        .sort(["outcome", "lever", "background", "value"])
    )


def interaction_table(summary: pl.DataFrame, outcome: str) -> pl.DataFrame:
    """`lever_effect` for one outcome, one row per (lever, value), one column
    per background — plus `swing`, the spread across backgrounds.

    `swing` is the interaction, and it is the number to sort on. A lever whose
    effect is the same under every background has swing ~0 and its screen row
    was telling the truth; a lever with a large swing does something different
    depending on the setting it is embedded in, and reporting its main effect
    alone is reporting an average over a factor it interacts with.

    Requires a summary built from `across=`-crossed cells; with a single
    background there is nothing to compare and `swing` is 0 by construction.
    A lever that is itself the background factor is omitted: it defines the
    facets rather than varying within them, and its main effect is in
    `summary` already.
    """
    frame = summary.filter(pl.col("outcome") == outcome)
    if frame.height == 0:
        raise ValueError(
            f"no rows for outcome {outcome!r}; available: "
            f"{sorted(summary['outcome'].unique())}"
        )

    backgrounds = set(frame["background"].unique())
    if len(backgrounds - {""}) > 1:
        # Cells with no background are the background factor's own values —
        # they define the facets, so they have no effect *within* one and
        # would pivot to a column of nulls. Their main effect is in `summary`.
        frame = frame.filter(pl.col("background") != "")

    wide = frame.pivot(on="background", index=["lever", "value"], values="lever_effect")
    background_cols = [c for c in wide.columns if c not in ("lever", "value")]
    return wide.with_columns(
        (pl.max_horizontal(background_cols) - pl.min_horizontal(background_cols)).alias("swing")
    ).sort("swing", descending=True, nulls_last=True)
