"""`across=`: crossing a screened lever against a background factor.

The screen is one-at-a-time, which means every lever is evaluated at the base
configuration. That is a weaker claim than it looks, and the cost is measurable:
`tau_position` reads flat in the screen and moves cross-camp exposure by +0.088
under `affinity` against -0.004 under `chronological`. These tests pin the
mechanics of the crossing; the interaction itself is pinned in
`test_attention_budget_binds.py`.
"""

from __future__ import annotations

import dataclasses

import polars as pl
import pytest

from discourse_lab.config import Config
from discourse_lab.data import scenario_config
from discourse_lab.experiments import (
    RANKER_BACKGROUND,
    build_interventions,
    interaction_table,
    run_interventions,
    summarize_interventions,
)

LEVERS = {
    "dynamics.ranker": ("chronological", "affinity"),
    "dynamics.tau_position": (6.0, 2.0),
    "dynamics.inject_k": (0, 20),
}


def _base(n_users=300, n_ticks=10):
    return scenario_config(dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=n_users, stance_dims=3),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=n_ticks, drift="none",
                                     exposure_sample_rate=0.10),
    ))


def test_screen_is_unchanged_when_no_background_is_given():
    cells = build_interventions(_base(), levers=LEVERS)
    assert len(cells) == sum(len(v) for v in LEVERS.values())
    assert all(cell.background == () and cell.background_label == "" for cell in cells)


def test_crossing_multiplies_the_non_background_levers_only():
    """The background factor is not crossed with itself: doing so would
    silently overwrite the background level with the lever value, and every
    facet would then be identical."""
    cells = build_interventions(_base(), levers=LEVERS, across=RANKER_BACKGROUND)

    crossed = [c for c in cells if c.background]
    main = [c for c in cells if not c.background]
    assert len(crossed) == 2 * (len(LEVERS["dynamics.tau_position"])
                                + len(LEVERS["dynamics.inject_k"]))
    # the ranker still contributes its own main effect, once per value
    assert sorted(str(c.value) for c in main) == ["affinity", "chronological"]
    assert all(c.lever == "dynamics.ranker" for c in main)


def test_every_crossed_cell_actually_carries_both_settings():
    """The bug this guards against is `set_param` order: applying the lever
    after the background is what makes the background stick."""
    cells = build_interventions(_base(), levers=LEVERS, across=RANKER_BACKGROUND)
    for cell in cells:
        if not cell.background:
            continue
        assert cell.cfg.dynamics.ranker == dict(cell.background)["dynamics.ranker"]
        field = cell.lever.split(".")[-1]
        assert getattr(cell.cfg.dynamics, field) == cell.value
        assert cell.null_cfg.dynamics.kernel == "null"


def test_configs_are_distinct_so_the_cache_does_not_collapse_cells():
    """Two cells that hash the same are one run reported twice."""
    cells = build_interventions(_base(), levers=LEVERS, across=RANKER_BACKGROUND)
    hashes = [c.cfg.hash() for c in cells]
    assert len(set(hashes)) == len(hashes)


def test_summary_keeps_backgrounds_apart_and_references_within_each():
    rows = run_interventions(
        build_interventions(_base(), levers=LEVERS, across=RANKER_BACKGROUND),
        seeds=[0], warn_on_few_seeds=False)
    summary = summarize_interventions(rows)

    assert set(summary["background"].unique()) == {
        "", "dynamics.ranker=chronological", "dynamics.ranker=affinity"}
    # The reference value has zero effect inside every background, not just
    # one. NaN outcomes are excluded rather than asserted on: `algorithmic_share`
    # is NaN at inject_k=0 by construction, and NaN - NaN is NaN, not 0.
    refs = summary.filter(
        (pl.col("lever") == "dynamics.tau_position")
        & (pl.col("value") == "6.0")
        & pl.col("lever_effect").is_not_nan())
    assert refs.height > 0
    assert (refs["lever_effect"].abs() < 1e-12).all()


def test_interaction_table_reports_the_swing_and_sorts_on_it():
    rows = run_interventions(
        build_interventions(_base(), levers=LEVERS, across=RANKER_BACKGROUND),
        seeds=[0], warn_on_few_seeds=False)
    table = interaction_table(summarize_interventions(rows),
                              "cross_cutting_exposure.camp_share")

    assert "swing" in table.columns
    assert "dynamics.ranker" not in table["lever"].to_list(), \
        "the background factor defines the facets; it cannot vary within them"
    swings = table["swing"].to_list()
    assert swings == sorted(swings, reverse=True)


def test_interaction_table_rejects_an_outcome_that_is_not_there():
    rows = run_interventions(build_interventions(_base(), levers=LEVERS),
                             seeds=[0], warn_on_few_seeds=False)
    with pytest.raises(ValueError, match="no rows for outcome"):
        interaction_table(summarize_interventions(rows), "not_an_outcome")
