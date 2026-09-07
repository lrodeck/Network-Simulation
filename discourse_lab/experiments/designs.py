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

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import polars as pl


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

    def run(self, base, seeds, metrics=None, persist=("posts", "engagements")) -> pl.DataFrame:
        """Run every grid cell AND its matched null, seed-major, returning
        tidy long-form with a `condition` (model|null) column.

        The null twin is derived HERE by replacing `dynamics.kernel` alone —
        the same contract `quality_attention_lift` asserts. A null that also
        swapped selection or affect would difference those mechanisms out of
        the result while the design text still said 'kernel'.

        The stylized gate runs first (two cheap seeds, pre-flight length)
        and warns on failure: a design's decision metric inherits §5.1's
        validity, so an off-gate base config announces itself before the
        sweep's budget is spent. Gate status carries no refusal — studying
        an off-gate configuration is legitimate; quoting one blindly is
        not."""
        from discourse_lab.analysis import expand_grid
        from discourse_lab.runner import cached_run, load_run

        from .gate import stylized_gate

        stylized_gate(base, seeds=(0, 1),
                      n_ticks=min(60, base.dynamics.n_ticks), warn=True)

        configs: list[tuple[dict, object, str]] = []
        for cell, cfg in expand_grid(base, self.grid):
            configs.append((cell, cfg, "model"))
            if "null" in self.controls:
                null_cfg = dataclasses.replace(
                    cfg, dynamics=dataclasses.replace(cfg.dynamics, kernel="null")
                )
                configs.append((cell, null_cfg, "null"))

        rows = []
        for s in seeds:                              # seed-major flat queue
            for cell, cfg, condition in configs:
                cached_run(cfg, s, persist=persist)
                frame = load_run(cfg, s).metrics()
                names = metrics or [c for c in frame.columns if c != "t"]
                for name in names:
                    if name not in frame.columns:
                        continue
                    values = frame[name].to_numpy().astype(float)
                    finite = values[np.isfinite(values)]
                    rows.append(
                        {**{k: str(v) for k, v in cell.items()},
                         "condition": condition, "seed": s, "metric": name,
                         "value": float(finite.mean()) if len(finite) else float("nan")}
                    )
        return pl.DataFrame(rows)


# The designs the change spec names, with their falsifiers stated up front.
NAMED_DESIGNS: dict[str, Design] = {
    # D3 (change spec): merit-driven attention, only meaningful against null.
    #
    # Budget, measured (17 us/user-tick; ~1.4 min per run at N=1e4, 500
    # ticks): the social_proof ladder was CUT to 4 levels and homophily
    # dropped from this design — it makes no competing prediction here, it
    # was just present, and it doubled the bill for nothing. 4 ladder levels
    # x (model + matched null) x 20 seeds = 160 runs ~= 3.8 h sequential,
    # ~30 min at 8 workers (DLAB_WORKERS=8). The earlier 24-cell version was
    # closer to a week of wall-clock for the same crossing point.
    #
    # The manipulation is a `theta_scale` LADDER over the social_proof group
    # (which carries prominence and tie_strength with it — the popularity
    # force broadly), on a kernel that rewards quality too. Swapping whole
    # kernels cannot find the crossing point: epistemic has no social_proof
    # to weaken and bandwagon has no quality to crowd out. The ladder is
    # what the group lever exists for.
    "d3_quality_attention": Design(
        name="d3_quality_attention",
        grid={
            "dynamics.kernel": ["bandwagon"],
            "dynamics.kernel_theta": [(
                ("like", "quality", 1.0), ("repost", "quality", 0.8),
            )],
            "dynamics.theta_scale": [
                (("social_proof", s),) for s in (0.0, 0.5, 1.0, 2.0)
            ],
        },
        controls=["null"],
        decision_metric="quality_attention_lift",
        predictions={
            "social_proof 0": "lift > 0 — merit drives attention",
            "social_proof 2": "lift -> 0 — popularity crowds merit out",
        },
        falsifier=(
            "the lift never falls as the social_proof ladder rises: a "
            "bandwagon force that cannot crowd merit out of attention, even "
            "at twice the shipped strength, means the model cannot produce "
            "the Salganik/Muchnik invisible-moderating effect it was built "
            "to interrogate"
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
