"""Figures and tables (dev §6 step 13).

Importing this module must never import matplotlib — `viz.tables` is
polars-only and useful on a headless box, and the core package must stay
lean. Figure symbols are resolved lazily through PEP 562 `__getattr__`, so
`from discourse_lab.viz import tables` costs nothing.
"""

from __future__ import annotations

_FIGURE_EXPORTS = {
    "fig_calibration": "discourse_lab.viz.figures.calibration",
    "fig_metric_trajectories": "discourse_lab.viz.figures.trajectory",
    "fig_engagement_ccdf": "discourse_lab.viz.figures.distribution",
    "fig_cascade_sizes": "discourse_lab.viz.figures.distribution",
    "fig_lorenz": "discourse_lab.viz.figures.distribution",
    "fig_effect_dots": "discourse_lab.viz.figures.experiment",
    "fig_degree_ccdf": "discourse_lab.viz.figures.network",
    "fig_archetypes": "discourse_lab.viz.figures.world",
    "fig_lever_effects": "discourse_lab.viz.figures.normative",
    "fig_scenario_axes": "discourse_lab.viz.figures.world",
    "fig_trait_correlations": "discourse_lab.viz.figures.world",
    "fig_contact_vs_hostility": "discourse_lab.viz.figures.normative",
    "save_figure": "discourse_lab.viz.save",
    "styled": "discourse_lab.viz.style",
    "series_colors": "discourse_lab.viz.style",
}

__all__ = [*sorted(_FIGURE_EXPORTS), "tables"]


def __getattr__(name: str):
    import importlib

    # import_module, never `from . import tables`: the from-form re-enters
    # this __getattr__ when the submodule is not yet in sys.modules, which
    # recurses until the stack gives out.
    if name == "tables":
        return importlib.import_module("discourse_lab.viz.tables")

    module_path = _FIGURE_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)
