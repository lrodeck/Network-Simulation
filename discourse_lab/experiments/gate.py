"""The stylized gate as a runnable check, not a one-off test (change spec
C9's closing move).

The C1-C10 changes moved the graph generator and the tick, so every
stylized-fact number previously calibrated was fitted under different
mechanisms. The recalibrated combination of record is
`calibrated_gate_config()`; this module lets ANY config be checked against
the gate, and the experiment runners call it before spending the sweep's
budget — so a Gini-quoting result carries its gate status with it instead of
inheriting one from a config that no longer passes.

V7.6 (change-spec-v7-continuous-affect.md) demoted `attention_gini` out of
the graded set: it is kernel-bound (in-band only under `bandwagon`; every
other kernel this codebase ships measures 0.6-0.75 or ~0.97 under `drift=
"full"`'s C3a compounding — see FINDINGS.md), so a config that legitimately
studies a different engagement theory can never pass it, and carrying it as
a pass/fail row produced three rounds of misleading "off-gate" language
before this. `GATE_ROWS` is `reciprocity` alone now; `attention_gini` (and
`clustering_ratio`, never graded here) are still measured and still
reported in `GateReport.rows` and the full `stylized_facts_from_run` table
— flagged against their reference range, just never blocking.

Warnings, not refusals: studying an off-gate configuration is legitimate —
quoting one without knowing it is not.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

from discourse_lab.analysis import set_param
from discourse_lab.config import Config
from discourse_lab.metrics import STYLIZED_FACT_RANGES, stylized_facts_from_run
from discourse_lab.network import cached_graph
from discourse_lab.population import cached_population
from discourse_lab.runner import cached_run, load_run, phase_rngs

GATE_ROWS = ("reciprocity",)


def calibrated_gate_config() -> Config:
    """The attention-inequality calibration of record: the configuration the
    20-seed C9 gate passes at (Gini 0.82-0.93, reciprocity 0.252-0.268 at
    N=1200 x 60 ticks). Start experiments that quote §5.1 bands from here,
    or run `stylized_gate` on your own base config and attach the report."""
    base = set_param(Config(), "dynamics.ranker", "engagement_optimized")
    base = set_param(base, "dynamics.kernel", "bandwagon")
    base = set_param(base, "graph.generator", "latent_pa")
    base = set_param(base, "graph.mirror_p", 0.05)
    base = set_param(base, "dynamics.theta_scale", (("social_proof", 0.6),))
    return base


@dataclass
class GateReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    rows: dict[str, list[float]] = field(default_factory=dict)

    def summary(self) -> str:
        head = ("PASSED" if self.passed else "FAILED") + (
            ": " + "; ".join(self.failures) if self.failures else ""
        )
        lines = [head]
        for name, values in self.rows.items():
            lines.append(
                f"  {name}: {min(values):.3f}-{max(values):.3f} over {len(values)} seeds"
            )
        return "\n".join(lines)


def stylized_gate(
    base: Config,
    seeds: Sequence[int] = (0, 1),
    n_ticks: int | None = None,
    warn: bool = False,
) -> GateReport:
    """Run the stylized-fact gate at `base`'s own settings, and grade the
    `GATE_ROWS` (V7.6: `reciprocity` alone — see module docstring) per seed.
    `n_ticks` scales a long config down for a cheap pre-flight. With
    `warn=True` a failure also emits a warning — the experiment runners set
    that, so an off-gate sweep announces itself before its numbers exist."""
    cfg = base
    if n_ticks is not None and cfg.dynamics.n_ticks > n_ticks:
        cfg = set_param(cfg, "dynamics.n_ticks", n_ticks)

    report = GateReport(passed=True)
    for seed in seeds:
        cached_run(cfg, seed, persist=("posts", "engagements"))
        handle = load_run(cfg, seed)
        rngs = phase_rngs(seed)
        graph = cached_graph(
            cfg, seed, cached_population(cfg, seed, rngs["population"]), rngs["graph"]
        )
        facts = stylized_facts_from_run(handle, graph=graph, pop=None)

        for name, entry in facts.items():
            value = entry.get("value")
            if not isinstance(value, (int, float)) or value != value:
                continue  # ungraded or unmeasurable rows carry no number
            report.rows.setdefault(name, []).append(float(value))

        for name in GATE_ROWS:
            entry = facts.get(name, {})
            lo, hi = STYLIZED_FACT_RANGES[name]
            value = entry.get("value")
            if not isinstance(value, (int, float)) or value != value:
                report.failures.append(
                    f"seed {seed}: {name} unmeasurable — the gate cannot pass on no data"
                )
            elif not (lo <= value <= hi):
                report.failures.append(
                    f"seed {seed}: {name} = {value:.3f} outside [{lo}, {hi}]"
                )

    report.passed = not report.failures
    if warn and not report.passed:
        warnings.warn(
            "stylized gate FAILED at this base config — §5.1's bands are the "
            "validity every §5.2/§5.3 comparison inherits. Failures:\n"
            + "\n".join(report.failures),
            stacklevel=2,
        )
    return report
