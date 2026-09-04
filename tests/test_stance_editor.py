"""Step 3 verification (dev §6): draw -> autosave -> load in Python -> sample
-> histogram matches the curve.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from discourse_lab.config import Config
from discourse_lab.population import sample_population
from discourse_lab.widgets.stance_editor import BINS, PRESETS, StanceEditorWidget, axes_to_scenario_json


def test_autosave_writes_scenario_compatible_json(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    widget = StanceEditorWidget(name="draft")

    path = widget.path()
    assert path.exists()  # constructor sets axes -> observer autosaves

    data = json.loads(path.read_text())
    axes = data["scenario"]["stance_axes"]
    assert len(axes) == 3
    assert all(ax["marginal"]["bins"] == BINS for ax in axes)
    assert all(len(ax["marginal"]["density"]) == BINS for ax in axes)


def test_draw_then_autosave_then_load_reflects_the_new_curve(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    widget = StanceEditorWidget(name="draft")

    # "draw" a polarized curve onto axis 0, as the JS front end would via update()
    new_axes = [dict(a) for a in widget.axes]
    new_axes[0] = {**new_axes[0], "density": list(PRESETS["polarized"]())}
    widget.axes = new_axes  # triggers the autosave observer

    reloaded = StanceEditorWidget.load("draft")
    # save()/load() renormalise, so the reloaded curve is the drawn one up to a
    # constant scale factor, not bit-identical.
    original = np.asarray(list(PRESETS["polarized"]()))
    saved = np.asarray(reloaded.axes[0]["density"])
    ratio = saved / original
    assert np.allclose(ratio, ratio[0], rtol=1e-3)


def test_drawn_curve_reproduces_in_the_sampled_population(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    widget = StanceEditorWidget(name="draft")
    new_axes = [dict(a) for a in widget.axes]
    new_axes[0] = {**new_axes[0], "density": list(PRESETS["polarized"]())}
    widget.axes = new_axes

    reloaded = StanceEditorWidget.load("draft")
    scenario = reloaded.to_scenario_config()
    cfg = dataclasses.replace(
        Config(),
        scenario=scenario,
        population=dataclasses.replace(Config().population, n_users=20_000),
    )
    pop = sample_population(cfg, np.random.default_rng(7))

    axis = scenario.stance_axes[0]
    stance = pop.X_used[:, pop.trait_names.index(f"stance_{axis['name']}")]

    lo, hi = axis["marginal"]["support"]
    edges = np.linspace(lo, hi, BINS + 1)
    hist_sample, _ = np.histogram(stance, bins=edges, density=True)
    target = np.asarray(axis["marginal"]["density"])

    coarse = 16
    step = BINS // coarse
    target_coarse = target[: coarse * step].reshape(coarse, step).mean(axis=1)
    sample_coarse = hist_sample[: coarse * step].reshape(coarse, step).mean(axis=1)
    corr = np.corrcoef(target_coarse, sample_coarse)[0, 1]
    assert corr > 0.9  # polarized (bimodal) shape survived draw -> save -> load -> sample


def test_axes_to_scenario_json_is_scale_invariant_on_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    widget = StanceEditorWidget(name="roundtrip")
    first = axes_to_scenario_json(widget.axes, "roundtrip")

    reloaded = StanceEditorWidget.load("roundtrip")
    second = axes_to_scenario_json(reloaded.axes, "roundtrip")

    for a, b in zip(first["scenario"]["stance_axes"], second["scenario"]["stance_axes"]):
        da, db = np.asarray(a["marginal"]["density"]), np.asarray(b["marginal"]["density"])
        ratio = db / da
        assert np.allclose(ratio, ratio[0], rtol=1e-3)
