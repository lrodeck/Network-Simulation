"""Run monitor anywidget (dev §6 step 9, §7.1): live measures while a run
executes, so a config's fate is visible within twenty ticks rather than only
at the end. Consumes `run_iter` directly — nothing here reads a completed
Run.

Rendered fields (spec §7.1 / dev §7.1): the salience/stance agreement pair,
bubble index, and attention Gini, plus cascade activity (`r_eff`) and raw
volume (`n_posts`, `n_exposures`, `n_engagements`).

Also a narrative panel, when the run is driven with `run_iter(..., narrate=True)`.
Six sparklines tell you a number moved; they do not tell you *what the
discourse is about*, which is the thing a normative study is watching for. The
sentence comes from `semantics.describe_state` and is deterministic — spec §0.1
bars an API call anywhere near the dynamics.
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
    narrative = traitlets.Unicode("").tag(sync=True)
    # capped: this syncs to the browser on every tick, and an uncapped list
    # would resend the whole run's narration each time
    narrative_log = traitlets.List(traitlets.Unicode()).tag(sync=True)

    NARRATIVE_LOG_LIMIT = 20

    def __init__(self, cfg=None, **kwargs):
        """`cfg` supplies the vocabulary. Without it the narrative panel stays
        empty even when summaries arrive — the sentence needs the scenario's
        axis and topic names to be worth reading."""
        super().__init__(series={name: [] for name in TRACKED_MEASURES}, **kwargs)
        self._lexicon = None
        if cfg is not None:
            from discourse_lab.semantics import lexicon_for

            self._lexicon = lexicon_for(cfg)

    def push(self, state: State) -> None:
        """Append one tick's metrics. Call this once per `run_iter` yield."""
        series = {k: list(v) for k, v in self.series.items()}
        for name in TRACKED_MEASURES:
            series.setdefault(name, []).append(float(state.metrics.get(name, float("nan"))))
        self.ticks = self.ticks + [state.t]
        self.series = series
        self.current_tick = state.t

        summary = getattr(state, "summary", None)
        if summary is not None and self._lexicon is not None:
            from discourse_lab.semantics import describe_state

            text = describe_state(summary, self._lexicon)
            self.narrative = text
            self.narrative_log = (self.narrative_log + [text])[-self.NARRATIVE_LOG_LIMIT :]

    def watch(self, states: Iterable[State]) -> None:
        """Drive the widget directly from a `run_iter(...)` generator.

        Pass `narrate=True` to `run_iter` and `cfg=` to this widget to get the
        narrative panel as well as the charts.
        """
        for state in states:
            self.push(state)
