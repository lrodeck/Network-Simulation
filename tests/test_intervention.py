"""The intervention study: design levers against normative outcomes.

The thing these guard is the contrast. It is easy to report the matched-null
delta as though it were the lever's effect, and for feed levers that number is
~0 by construction because the null holds the lever fixed.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import polars as pl
import pytest

from discourse_lab.config import Config
from discourse_lab.data import scenario_config
from discourse_lab.experiments import (
    DEFAULT_LEVERS,
    build_interventions,
    run_interventions,
    summarize_interventions,
)


def _base(n_users=400, n_ticks=12):
    return scenario_config(
        dataclasses.replace(
            Config(),
            population=dataclasses.replace(Config().population, n_users=n_users, stance_dims=3),
            dynamics=dataclasses.replace(
                Config().dynamics, n_ticks=n_ticks, drift="none", exposure_sample_rate=0.10
            ),
        )
    )


def _sweep(tmp_path, monkeypatch, levers, seeds=3):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cells = build_interventions(_base(), levers=levers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = run_interventions(cells, seeds=range(seeds), warn_on_few_seeds=False)
    return rows, summarize_interventions(rows)


def test_every_cell_gets_a_null_that_differs_only_in_the_kernel():
    """spec §5.3's protocol: the null must match on the lever, or the contrast
    confounds the lever with the kernel."""
    cells = build_interventions(_base(), levers={"dynamics.inject_k": (0, 7)})
    for cell in cells:
        assert cell.null_cfg.dynamics.kernel == "null"
        assert cell.cfg.dynamics.kernel != "null"
        assert cell.null_cfg.graph == cell.cfg.graph
        assert cell.null_cfg.population == cell.cfg.population
        # the lever itself is held identical across the pair
        assert cell.null_cfg.dynamics.inject_k == cell.cfg.dynamics.inject_k


def test_one_at_a_time_not_a_factorial():
    """spec §5.4 asks for one-at-a-time. Four levers factorially would be 108
    cells against 13; interactions are a follow-up, not the first result."""
    cells = build_interventions(_base())
    assert len(cells) == sum(len(v) for v in DEFAULT_LEVERS.values())


def test_lever_effect_is_measured_against_the_levers_reference_value(tmp_path, monkeypatch):
    """The headline contrast is between lever values, not against the null.

    Measured: the ranker moves cross-camp exposure by ~0.078 between values
    while its kernel_delta is ~-0.004, because the matched null carries the
    same feed effect. Reporting the delta as the lever's effect would say the
    ranker does nothing.
    """
    rows, summary = _sweep(
        tmp_path, monkeypatch, {"dynamics.ranker": ("chronological", "affinity")}
    )
    camp = summary.filter(pl.col("outcome") == "cross_cutting_exposure.camp_share")

    reference = camp.filter(pl.col("value") == "chronological")
    treated = camp.filter(pl.col("value") == "affinity")
    assert reference["lever_effect"][0] == 0.0, "the reference value is its own baseline"
    assert treated["lever_effect"][0] < 0, "a bubble-maximal ranker must lower cross-cutting"
    assert abs(treated["lever_effect"][0]) > abs(treated["kernel_delta"][0])


def test_summary_reports_resolution_rather_than_filtering_on_it(tmp_path, monkeypatch):
    """"This design choice does not move this outcome at a sample size a study
    can afford" is a result, so unresolved rows stay in the frame."""
    rows, summary = _sweep(tmp_path, monkeypatch, {"dynamics.inject_k": (0, 20)})
    assert "resolves" in summary.columns
    assert summary["resolves"].dtype == pl.Boolean
    assert len(summary.filter(~pl.col("resolves"))) > 0, "nothing unresolved to report on"


def test_rows_are_tidy_long_form(tmp_path, monkeypatch):
    rows, _ = _sweep(tmp_path, monkeypatch, {"dynamics.inject_k": (0, 20)}, seeds=2)
    assert set(rows.columns) == {
        "lever", "value", "seed", "outcome", "model", "null", "kernel_delta"
    }
    # bookkeeping fields are not outcomes
    assert not any(o.endswith(".n") for o in rows["outcome"].unique())


def test_reference_value_can_be_chosen(tmp_path, monkeypatch):
    rows, _ = _sweep(tmp_path, monkeypatch, {"dynamics.ranker": ("chronological", "affinity")})
    flipped = summarize_interventions(rows, reference={"dynamics.ranker": "affinity"})
    camp = flipped.filter(pl.col("outcome") == "cross_cutting_exposure.camp_share")
    assert camp.filter(pl.col("value") == "affinity")["lever_effect"][0] == 0.0
    assert camp.filter(pl.col("value") == "chronological")["lever_effect"][0] > 0


def test_few_seeds_warns():
    """spec §4.4 calls too few seeds the single most common way simulation
    studies of this kind go wrong."""
    cells = build_interventions(_base(n_users=200, n_ticks=4), levers={"dynamics.inject_k": (0,)})
    with pytest.warns(UserWarning, match="at least"):
        run_interventions(cells, seeds=[0])
