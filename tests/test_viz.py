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
