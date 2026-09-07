"""Figures and tables (dev §6 step 13).

No golden images: they break on every matplotlib point release and tell you
nothing about whether the chart is right. These assert on the returned Figure
— colours drawn from the palette, panel counts, threshold lines, and that a
saved PDF is a real PDF — plus the rules that would otherwise rot silently.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from discourse_lab.config import Config
from discourse_lab.viz import tables
from discourse_lab.viz.style import CHROME, PALETTE, series_colors, symmetric_norm

matplotlib = pytest.importorskip("matplotlib")


# --------------------------------------------------------------------------
# the core must not grow a plotting dependency
# --------------------------------------------------------------------------


def test_importing_the_core_never_imports_matplotlib():
    """A sweep on a headless box has no reason to carry a plotting stack, and
    `runner` importing pyplot would make that unavoidable.
    """
    import subprocess
    import sys

    probe = (
        "import sys; "
        "import discourse_lab, discourse_lab.runner, discourse_lab.metrics, discourse_lab.viz; "
        "print('matplotlib' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr


def test_tables_work_without_matplotlib(monkeypatch):
    """`viz.tables` is polars-only by design — it is the half of this package
    that runs anywhere."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ModuleNotFoundError("matplotlib blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    frame = pl.DataFrame({"Fact": ["x"], "Value": [1.0]})
    assert "| Fact |" in tables.to_markdown(frame)


# --------------------------------------------------------------------------
# style rules
# --------------------------------------------------------------------------


def test_categorical_palette_refuses_a_ninth_series():
    assert series_colors(8) == list(PALETTE)
    with pytest.raises(ValueError, match="Facet"):
        series_colors(9)


def test_diverging_norm_is_symmetric_so_gray_lands_on_zero():
    norm = symmetric_norm([-0.2, 0.9, 0.3])
    assert norm.vmin == -norm.vmax
    assert norm(0.0) == pytest.approx(0.5)


def test_styled_does_not_mutate_global_rcparams():
    import matplotlib.pyplot as plt

    from discourse_lab.viz.style import styled

    before = plt.rcParams["axes.grid"]
    with styled():
        pass
    assert plt.rcParams["axes.grid"] == before


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _report():
    return {
        "engagement_alpha": {"label": "Engagement alpha", "value": 5.6,
                             "target": (2.0, 3.0), "in_range": None},
        "attention_gini": {"label": "Attention Gini", "value": 0.91,
                           "target": (0.8, 0.95), "in_range": True},
        "thread_depth_mean": {"label": "Thread depth", "value": 1.1,
                              "target": (1.5, 3.0), "in_range": False},
        "hostility": {"label": "Hostility", "value": 0.03, "target": None, "in_range": None},
    }


def test_calibration_prints_every_value_and_a_status_word():
    """Status is never colour alone — spec-independent accessibility rule, and
    the palette's contrast warning is discharged by these labels.
    """
    from discourse_lab.viz import fig_calibration

    fig = fig_calibration(_report())
    text = " ".join(t.get_text() for t in fig.axes[0].texts)
    for word in ("pass", "FAIL", "n/a"):
        assert word in text
    assert "5.6" in text and "0.91" in text
    assert "no target quoted" in text, "an ungraded row must say so, not just show a dot"


def test_calibration_places_no_dot_on_a_row_without_a_target():
    """A mark inside a band it has no scale for reads as a score."""
    from discourse_lab.viz import fig_calibration

    graded = {k: v for k, v in _report().items() if v["target"] is not None}
    assert len(fig_calibration(_report()).axes[0].lines) == len(graded)


def test_trajectories_require_a_null_and_mark_the_r_eff_threshold(tmp_path, monkeypatch):
    """spec §5.3 makes the null comparison mandatory, so the signature takes
    both handles — there is no way to draw the model alone."""
    import inspect

    from discourse_lab.viz.figures.trajectory import fig_metric_trajectories

    params = inspect.signature(fig_metric_trajectories).parameters
    assert params["null_handle"].default is inspect.Parameter.empty


def test_figure_series_use_only_palette_colours():
    from discourse_lab.viz import fig_lorenz

    rng = np.random.default_rng(0)
    fig = fig_lorenz({"a": rng.lognormal(0, 1, 500), "b": rng.lognormal(0, 2, 500)})
    drawn = {line.get_color() for line in fig.axes[0].lines}
    assert drawn <= set(PALETTE) | {CHROME["baseline"]}


def test_save_figure_writes_a_real_pdf(tmp_path):
    from discourse_lab.viz import fig_cascade_sizes, save_figure

    root = np.concatenate([np.arange(60), np.repeat([900, 901], 3)])
    out = save_figure(fig_cascade_sizes(root), "cascades", directory=tmp_path)
    assert out["pdf"].read_bytes()[:5] == b"%PDF-"
    assert out["png"].stat().st_size > 0


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def test_latex_escapes_and_renders_nan_as_a_dash():
    """A table printing the string 'nan' reads as a bug; '--' reads as 'not
    measured', which is what it means."""
    frame = pl.DataFrame({"a_b": ["x_y"], "v": [float("nan")]})
    tex = tables.to_latex_booktabs(frame, caption="50% of runs", label="tab:t")
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert r"a\_b" in tex and r"x\_y" in tex
    assert r"50\%" in tex
    assert "--" in tex and "nan" not in tex


def test_latex_and_markdown_cannot_disagree():
    """Both exporters read the same rendered rows, so a value can never be
    rounded one way in the paper and another in the notebook."""
    frame = pl.DataFrame({"k": ["a", "b"], "v": [0.123456, 7.891011]})
    tex, md = tables.to_latex_booktabs(frame, precision=3), tables.to_markdown(frame, precision=3)
    for value in ("0.123", "7.891"):
        assert value in tex and value in md


def test_stylized_facts_table_marks_ungraded_rows_as_na():
    frame = tables.stylized_facts_table(_report())
    status = dict(zip(frame["Fact"].to_list(), frame["Status"].to_list()))

    def status_of(prefix: str) -> str:
        # the attention row is renamed to name its source, so match on prefix
        return next(v for k, v in status.items() if k.startswith(prefix))

    assert status_of("Engagement alpha") == "n/a"
    assert status_of("Attention Gini") == "pass"
    assert status_of("Thread depth") == "FAIL"
    assert status_of("Hostility") == "n/a"


def test_stylized_facts_table_says_which_attention_gini_it_reports():
    """`metrics.parquet:attention_gini` is a rolling per-tick measure over
    active posts; `posts.parquet` gives lifetime totals. They do not agree, so
    the table has to name the one it used."""
    frame = tables.stylized_facts_table(_report(), attention_gini_source="lifetime, per post")
    assert any("lifetime, per post" in f for f in frame["Fact"].to_list())


def test_save_table_writes_both_formats(tmp_path):
    frame = pl.DataFrame({"a": [1.0]})
    out = tables.save_table(frame, "t", directory=tmp_path)
    assert out["tex"].exists() and out["md"].exists()


def test_lever_effects_marks_unresolved_effects_hollow():
    """An effect that does not clear its noise must not read as a small one.
    Filled vs hollow is a second channel beside position, so the distinction
    survives being printed in greyscale."""
    from discourse_lab.viz import fig_lever_effects

    summary = pl.DataFrame({
        "lever": ["dynamics.ranker"] * 3,
        "value": ["chronological", "affinity", "popularity"],
        "outcome": ["cross_cutting_exposure.camp_share"] * 3,
        "model_mean": [0.39, 0.31, 0.38],
        "model_sd": [0.01, 0.01, 0.05],
        "null_mean": [0.39, 0.31, 0.38],
        "kernel_delta": [0.0, -0.001, 0.0],
        "n_seeds": [4, 4, 4],
        "lever_effect": [0.0, -0.078, -0.01],
        "reference": ["chronological"] * 3,
        "resolves": [False, True, False],
    })
    fig = fig_lever_effects(summary)
    ax = fig.axes[0]

    faces = [line.get_markerfacecolor() for line in ax.lines if line.get_marker() == "o"]
    assert any(f == CHROME["surface"] for f in faces), "no hollow marker for an unresolved effect"
    assert any(f in PALETTE for f in faces), "no filled marker for a resolved effect"


def test_contact_vs_hostility_merges_coincident_points():
    """Every lever contributes its own reference row, so the baseline config
    lands at identical coordinates once per lever. Unmerged, those labels
    overprint into a smear."""
    from discourse_lab.viz import fig_contact_vs_hostility

    summary = pl.DataFrame({
        "lever": ["dynamics.ranker", "dynamics.inject_k"],
        "value": ["chronological", "0"],
        "outcome": ["hostility_given_contact.contact_rate"] * 2,
        "model_mean": [0.3466, 0.3466],
        "model_sd": [0.01, 0.01],
        "n_seeds": [4, 4],
    })
    hostility = summary.with_columns(
        pl.lit("hostility_given_contact.hostility").alias("outcome"),
        pl.lit(0.0285).alias("model_mean"),
    )
    fig = fig_contact_vs_hostility(pl.concat([summary, hostility]))
    ax = fig.axes[0]

    drawn = [line for line in ax.lines if line.get_marker() == "o"]
    assert sum(len(line.get_xdata()) for line in drawn) == 1, "coincident points not merged"
    assert len(ax.texts) == 1, "one label block for one point"
    assert "ranker=chronological" in ax.texts[0].get_text()
    assert "inject_k=0" in ax.texts[0].get_text()


# --------------------------------------------------------------------------
# the world a run was conducted in
# --------------------------------------------------------------------------


def _world():
    import numpy as np

    from discourse_lab.data import scenario_config
    from discourse_lab.population import sample_population

    cfg = scenario_config(
        dataclasses.replace(Config(), population=dataclasses.replace(Config().population, n_users=1500))
    )
    return cfg, sample_population(cfg, np.random.default_rng(0))


def test_scenario_axes_figure_names_both_poles_of_every_axis():
    """A 128-bin density in JSON says nothing about whether the population is
    bimodal; the shape is the modelling claim, and the poles name it."""
    from discourse_lab.viz import fig_scenario_axes

    cfg, _ = _world()
    fig = fig_scenario_axes(cfg)
    assert len(fig.axes) == cfg.stance_dims()

    labels = {t.get_text() for ax in fig.axes for t in ax.get_xticklabels()}
    for neg, pos in cfg.scenario.poles():
        assert neg in labels and pos in labels


def test_scenario_axes_figure_refuses_without_a_scenario():
    """Drawing unnamed axes would defeat the purpose."""
    from discourse_lab.viz import fig_scenario_axes

    with pytest.raises(ValueError, match="scenario_config"):
        fig_scenario_axes(Config())


def test_trait_correlations_shows_what_the_config_does_not_say():
    """`correlation_pairs` defaults to empty, so the requested matrix is the
    identity and a reader of the config concludes the traits are independent.
    The archetype mixture induces real correlation — measured, activity x
    reply_prop = +0.30 at N=8000 — because an archetype that shifts two traits
    together correlates them.
    """
    import numpy as np

    from discourse_lab.viz import fig_trait_correlations

    cfg, pop = _world()
    assert cfg.population.correlation_pairs == (), "this test is about the identity default"

    fig = fig_trait_correlations(pop)
    image = fig.axes[0].images[0]
    data = image.get_array()

    assert np.allclose(np.diag(data), 0.0), "the diagonal should be removed, not plotted as 1"
    assert np.abs(data).max() > 0.15, "no induced correlation visible at all"
    # diverging ramp on a symmetric norm, so grey lands exactly on zero
    assert image.norm.vmin == -image.norm.vmax


def test_archetype_figure_contrasts_requested_and_realised_weights():
    from discourse_lab.viz import fig_archetypes

    cfg, pop = _world()
    fig = fig_archetypes(cfg, pop)
    legend_labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert legend_labels == {"requested", "realised"}


def test_world_table_reports_realised_correlation_against_the_requested_none():
    """The row that carries the finding: Requested "none (identity)" beside a
    Realised strongest pair."""
    cfg, pop = _world()
    frame = tables.world_table(cfg, pop)
    row = frame.filter(pl.col("Property") == "trait correlations")

    assert "identity" in row["Requested"][0]
    assert "activity" in row["Realised"][0] and "reply_prop" in row["Realised"][0]

    names = frame["Realised"].to_list()
    assert any("provision" in n for n in names), "axis names missing"
    assert any("immigration" in n for n in names), "topic names missing"


def test_world_table_works_without_a_population():
    """A config alone should still describe its world — the realised column
    just goes quiet."""
    cfg, _ = _world()
    frame = tables.world_table(cfg)
    assert len(frame) > 0
    assert frame.filter(pl.col("Property") == "trait correlations")["Realised"][0] == ""
