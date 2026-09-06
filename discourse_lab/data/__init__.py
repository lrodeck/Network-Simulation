"""Scenarios that ship with the package.

`discourse_lab/data/scenarios/*.json` has always been package data and nothing
could load it: `io.workspace.scenarios_dir()` points at the *workspace*
(`$DLAB_HOME/dlab/scenarios`), which is where the stance-editor widget saves,
not where the shipped files live. Attaching the default scenario to a Config
meant hand-rolling `json.load` + `ScenarioConfig.from_editor_json` +
`dataclasses.replace`, which only the tests ever did.
"""

from __future__ import annotations

import dataclasses
import json
from importlib import resources

from discourse_lab.config import Config, ScenarioConfig

_PACKAGE = "discourse_lab.data.scenarios"


def packaged_scenarios() -> list[str]:
    """Names of the scenarios shipped with the package."""
    return sorted(
        p.name[: -len(".json")]
        for p in resources.files(_PACKAGE).iterdir()
        if p.name.endswith(".json")
    )


def load_scenario_json(name: str = "default") -> dict:
    path = resources.files(_PACKAGE) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no packaged scenario {name!r}; available: {packaged_scenarios()}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_scenario(name: str = "default") -> ScenarioConfig:
    return ScenarioConfig.from_editor_json(load_scenario_json(name), name=name)


def scenario_config(
    base: Config | None = None, name: str = "default", allow_dim_change: bool = False
) -> Config:
    """Attach a packaged scenario to a config.

    Refuses by default when the scenario's axis count differs from
    `population.stance_dims`. A scenario silently overrides that field through
    `Config.stance_dims()`, and stance dimensionality is not a cosmetic
    setting: measured on this model, the homophily kernel's agreement effect
    against its matched null is +0.0049 (t=+2.46) at D=1 and -0.0001
    (t=-0.05) at D=3, because agreement requires simultaneous alignment on
    every axis. Swapping a scenario can therefore silently remove the
    mechanism a study is about. Pass `allow_dim_change=True` when the change
    is the point.
    """
    base = base if base is not None else Config()
    scenario = load_scenario(name)

    declared = base.population.stance_dims
    if not allow_dim_change and declared > 0 and declared != scenario.axis_count():
        raise ValueError(
            f"scenario {name!r} has {scenario.axis_count()} stance axes but "
            f"population.stance_dims is {declared}; the scenario wins, which "
            f"changes D from {declared} to {scenario.axis_count()} and with it "
            "which mechanisms are measurable. Pass allow_dim_change=True if "
            "that is intended."
        )
    return dataclasses.replace(base, scenario=scenario)
