"""Discourse Lab — a vectorized, non-agentic simulation toolbox for
theories of large-scale online discourse (spec §0).

Dynamics are numeric; language is a rendering pass that never runs here.
"""

from discourse_lab.config import (
    Config,
    DynamicsConfig,
    GraphConfig,
    PopulationConfig,
    ScenarioConfig,
)
from discourse_lab.runner import cached_run, load_run, phase_rngs, run, run_iter
from discourse_lab.sweep import sweep

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DynamicsConfig",
    "GraphConfig",
    "PopulationConfig",
    "ScenarioConfig",
    "run",
    "run_iter",
    "cached_run",
    "load_run",
    "sweep",
    "phase_rngs",
    "__version__",
]
