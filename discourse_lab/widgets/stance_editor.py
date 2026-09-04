"""Stance editor as an anywidget (dev §6 step 3), ported from `stance-editor.jsx`.

Written as plain DOM/canvas JS rather than React specifically so there is no
build step: Colab cannot run esbuild at install time, and hand-written vanilla
JS needs no bundling — the file in `_static/` *is* the shipped artifact.

Widget state is backed by files: every change autosaves to
`scenarios/<name>.json` in exactly the schema `ScenarioConfig.from_editor_json`
reads, so `StanceEditorWidget.load(name)` and `sample_population` close the
loop from drawn curve to sampled population.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anywidget
import traitlets

from discourse_lab.config import ScenarioConfig
from discourse_lab.io.workspace import scenarios_dir

BINS = 128
_STATIC = Path(__file__).parent / "_static"


def _gauss_mix(components: list[tuple[float, float, float]]) -> list[float]:
    import math

    d = [0.0] * BINS
    for i in range(BINS):
        x = -1 + (2 * (i + 0.5)) / BINS
        v = 0.0
        for mu, sd, w in components:
            v += (w * math.exp(-((x - mu) ** 2) / (2 * sd * sd))) / sd
        d[i] = v
    m = max(d) or 1.0
    return [v / m for v in d]


PRESETS: dict[str, Any] = {
    "symmetric": lambda: _gauss_mix([(0, 0.34, 1)]),
    "majority left": lambda: _gauss_mix([(-0.42, 0.32, 1), (0.55, 0.22, 0.28)]),
    "polarized": lambda: _gauss_mix([(-0.6, 0.2, 1), (0.6, 0.2, 1)]),
    "skewed tail": lambda: _gauss_mix([(-0.55, 0.22, 1), (0.1, 0.45, 0.45)]),
    "flat": lambda: [0.7] * BINS,
}


def default_axes() -> list[dict]:
    """The same three starter axes as the original React component."""
    return [
        {
            "id": 0,
            "name": "provision",
            "pole_neg": "market",
            "pole_pos": "state",
            "density": PRESETS["majority left"](),
            "floor": 0.004,
            "cost_neg": 0.0,
            "cost_pos": 0.35,
        },
        {
            "id": 1,
            "name": "openness",
            "pole_neg": "closed",
            "pole_pos": "open",
            "density": PRESETS["polarized"](),
            "floor": 0.004,
            "cost_neg": 0.3,
            "cost_pos": 0.1,
        },
        {
            "id": 2,
            "name": "institutional trust",
            "pole_neg": "distrust",
            "pole_pos": "trust",
            "density": PRESETS["symmetric"](),
            "floor": 0.004,
            "cost_neg": 0.0,
            "cost_pos": 0.0,
        },
    ]


def _normalise(density: list[float], floor: float) -> list[float]:
    p = [max(v, floor) for v in density]
    total = sum(p)
    if total <= 0:
        return [1.0 / BINS] * BINS
    return [v / total for v in p]


def axes_to_scenario_json(axes: list[dict], name: str) -> dict:
    """Exactly the schema the stance editor emits and `ScenarioConfig.from_editor_json` reads."""
    stance_axes = []
    for a in axes:
        p = _normalise(a["density"], a["floor"])
        stance_axes.append(
            {
                "name": a["name"],
                "pole_neg": a["pole_neg"],
                "pole_pos": a["pole_pos"],
                "marginal": {
                    "kind": "empirical",
                    "bins": BINS,
                    "support": [-1, 1],
                    "density": [round(v * BINS, 5) for v in p],
                },
                "expression_cost": {"neg": round(a["cost_neg"], 3), "pos": round(a["cost_pos"], 3)},
            }
        )
    return {"scenario": {"stance_axes": stance_axes}}


class StanceEditorWidget(anywidget.AnyWidget):
    """Draw density curves per stance axis; autosaves to scenarios/<name>.json."""

    _esm = _STATIC / "stance_editor.js"
    _css = _STATIC / "stance_editor.css"

    name = traitlets.Unicode("scenario").tag(sync=True)
    axes = traitlets.List(traitlets.Dict()).tag(sync=True)
    seed = traitlets.Int(12345).tag(sync=True)
    show_samples = traitlets.Bool(True).tag(sync=True)
    autosave = traitlets.Bool(True)

    def __init__(self, name: str = "scenario", axes: list[dict] | None = None, **kwargs):
        super().__init__(name=name, axes=axes if axes is not None else default_axes(), **kwargs)
        self.observe(self._on_axes_change, names=["axes"])
        if self.autosave:
            self.save()  # the constructor's own axes assignment predates the observer

    def _on_axes_change(self, change) -> None:
        if self.autosave:
            self.save()

    def path(self) -> Path:
        return scenarios_dir() / f"{self.name}.json"

    def save(self, path: Path | None = None) -> Path:
        path = path or self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(axes_to_scenario_json(self.axes, self.name), indent=2), encoding="utf-8")
        return path

    def to_scenario_config(self) -> ScenarioConfig:
        data = axes_to_scenario_json(self.axes, self.name)
        return ScenarioConfig.from_editor_json(data, name=self.name)

    @classmethod
    def load(cls, name: str, path: Path | None = None) -> "StanceEditorWidget":
        path = path or (scenarios_dir() / f"{name}.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        axes = []
        for i, ax in enumerate(data["scenario"]["stance_axes"]):
            m = ax["marginal"]
            cost = ax.get("expression_cost", {"neg": 0.0, "pos": 0.0})
            axes.append(
                {
                    "id": i,
                    "name": ax["name"],
                    "pole_neg": ax.get("pole_neg", ""),
                    "pole_pos": ax.get("pole_pos", ""),
                    "density": list(m["density"]),
                    "floor": 0.0,
                    "cost_neg": cost.get("neg", 0.0),
                    "cost_pos": cost.get("pos", 0.0),
                }
            )
        widget = cls(name=name, axes=axes)
        return widget
