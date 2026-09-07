"""C10.3 — the discriminating experiments as named, runnable designs.

A Design declares its manipulation, its controls (which must include a null
condition — §5.3's protocol is not optional), its decision metric, the
predicted direction per theory, and — required, no default — its falsifier.
A design that cannot state what would sink it does not get to run; the
required field is the point.

Predictions are strings, not numbers, because the claims are directional
("rise", "fall", "crossover", "no change"): the numeric threshold lives in
the decision metric and in the falsifier, where it can be argued about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from discourse_lab.analysis import sweep


@dataclass(frozen=True)
class Design:
    name: str
    grid: dict[str, list]
    controls: list[str]              # must include a null condition
    decision_metric: str
    predictions: dict[str, str]      # theory -> predicted direction
    falsifier: str                   # required; no default

    def __post_init__(self):
        if not self.falsifier:
            raise ValueError(
                f"design {self.name!r} has no falsifier: a design that cannot "
                "state what would sink it does not get to run (change spec C10.3)"
            )
        if "null" not in self.controls:
            raise ValueError(
                f"design {self.name!r} lacks a null control — §5.3: every "
                "reported effect is a difference against a matched null"
            )

    def run(self, base, seeds, metrics=None) -> pl.DataFrame:
        return sweep(base, self.grid, seeds, metrics=metrics)


# The designs the change spec names, with their falsifiers stated up front.
NAMED_DESIGNS: dict[str, Design] = {
    # D3 (change spec): merit-driven attention, only meaningful against null
    "d3_quality_attention": Design(
        name="d3_quality_attention",
        grid={"dynamics.kernel": ["epistemic", "bandwagon", "null"]},
        controls=["null"],
        decision_metric="quality_attention_lift",
        predictions={"epistemic": "rise", "bandwagon": "no change"},
        falsifier=(
            "epistemic's quality_attention_lift is not positive under "
            "quality_trait_coupling=0 across seeds — merit does not drive "
            "attention even when the kernel is authored to reward it"
        ),
    ),
    # D4: the Bakshy decomposition — choice vs algorithm
    "d4_choice_vs_algorithm": Design(
        name="d4_choice_vs_algorithm",
        grid={
            "dynamics.selection": ["position_only", "homophilous"],
            "dynamics.ranker": ["chronological", "engagement_optimized"],
        },
        controls=["null"],
        decision_metric="selection_filtering.selection_shift",
        predictions={
            "homophilous": "rise",
            "position_only": "no change",
        },
        falsifier=(
            "homophilous selection's echo-attended minus echo-exposed shift "
            "is indistinguishable from zero while cross-cutting exposure "
            "still moves with the ranker — choice would then be filtering "
            "less than the algorithm, contra Bakshy et al. (2015)"
        ),
    ),
    # D5: simple vs complex contagion crossover
    "d5_contagion_crossover": Design(
        name="d5_contagion_crossover",
        grid={
            "dynamics.reply_model": ["hawkes", "threshold"],
            "graph.generator": ["sbm", "configuration_model"],
        },
        controls=["null"],
        decision_metric="n_replies",
        predictions={
            "hawkes": "faster spread across long ties (configuration)",
            "threshold": "faster spread inside high clustering (sbm)",
        },
        falsifier=(
            "no reply_model x graph-structure interaction: both contagion "
            "models respond identically to clustering, so `threshold` is "
            "not actually complex — first check the distinct-neighbour "
            "count is distinct"
        ),
    ),
    # D6: bipolarization without repulsion
    "d6_repulsion_free_polarization": Design(
        name="d6_repulsion_free_polarization",
        grid={
            "dynamics.repulsion": [True, False],
            "dynamics.kernel": ["outrage", "null"],
        },
        controls=["null"],
        decision_metric="camp_bimodality",
        predictions={
            "outrage + C3 reinforcement": "bipolarization without repulsion",
            "repulsion on": "bipolarization",
        },
        falsifier=(
            "bipolarization (Sarle > 5/9) fails to emerge under every "
            "kernel with repulsion=False — the model inherits the "
            "literature's dependency on the repulsive-influence assumption "
            "(Mäs & Flache 2013; Takács et al. 2016), and MODEL.md §15 "
            "must say so plainly"
        ),
    ),
}


def design(name: str) -> Design:
    return NAMED_DESIGNS[name]


def design_names() -> list[str]:
    return sorted(NAMED_DESIGNS)
