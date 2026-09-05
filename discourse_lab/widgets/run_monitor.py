"""Run monitor anywidget (dev §6 step 9, §7.1): live measures while a run
executes, so a config's fate is visible within twenty ticks rather than only
at the end. Consumes `run_iter` directly — nothing here reads a completed
Run.

Rendered fields (spec §7.1 / dev §7.1): the salience/stance agreement pair,
bubble index, and attention Gini, plus cascade activity (`r_eff`) and raw
volume (`n_posts`, `n_exposures`, `n_engagements`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import anywidget
import traitlets

from discourse_lab.runner import State

_STATIC = Path(__file__).parent / "_static"

TRACKED_MEASURES = (
    "attention_gini",
    "salience",
    "agreement",
    "bubble_index",
    "r_eff",
    "n_posts",
    "n_exposures",
    "n_engagements",
)


class RunMonitorWidget(anywidget.AnyWidget):
    _esm = _STATIC / "run_monitor.js"
    _css = _STATIC / "run_monitor.css"

    ticks = traitlets.List(traitlets.Int()).tag(sync=True)
    series = traitlets.Dict().tag(sync=True)  # {measure_name: [values...]}
    current_tick = traitlets.Int(-1).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(series={name: [] for name in TRACKED_MEASURES}, **kwargs)

    def push(self, state: State) -> None:
        """Append one tick's metrics. Call this once per `run_iter` yield."""
        series = {k: list(v) for k, v in self.series.items()}
        for name in TRACKED_MEASURES:
            series.setdefault(name, []).append(float(state.metrics.get(name, float("nan"))))
        self.ticks = self.ticks + [state.t]
        self.series = series
        self.current_tick = state.t

    def watch(self, states: Iterable[State]) -> None:
        """Drive the widget directly from a `run_iter(...)` generator."""
        for state in states:
            self.push(state)
