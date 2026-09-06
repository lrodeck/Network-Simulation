"""The deterministic narrator.

No golden strings: the wording will change and asserting on it would make
every improvement a test failure. These assert on behaviour — determinism,
vocabulary membership, and that the numbers move in the direction the
mechanism predicts.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.data import scenario_config
from discourse_lab.runner import run_iter
from discourse_lab.semantics import describe_state, lexicon_for
from discourse_lab.semantics.narrate import BIMODAL_THRESHOLD, Narrator


def _cfg(n_users=600, n_ticks=15, **dyn):
    return scenario_config(
        dataclasses.replace(
            Config(),
            population=dataclasses.replace(Config().population, n_users=n_users, stance_dims=3),
            dynamics=dataclasses.replace(Config().dynamics, n_ticks=n_ticks, drift="none", **dyn),
        )
    )


def _last_summary(cfg, seed=0):
    return list(run_iter(cfg, seed=seed, narrate=True))[-1].summary


def test_narration_is_deterministic():
    """spec §0.1 bars an API call in dynamics, and §3 makes a run a pure
    function of (Config, seed) — narration must not break either."""
    cfg = _cfg()
    lex = lexicon_for(cfg)
    first = describe_state(_last_summary(cfg), lex)
    second = describe_state(_last_summary(cfg), lex)
    assert first == second
    assert len(first) > 0


def test_narration_only_uses_names_the_lexicon_knows():
    """Set membership, not text matching — the sentence structure is free to
    change but an invented topic or pole name is a bug."""
    cfg = _cfg()
    lex = lexicon_for(cfg)
    text = describe_state(_last_summary(cfg), lex)

    vocabulary = set(lex.topic_names) | set(lex.axis_names)
    for neg, pos in lex.axis_poles:
        vocabulary |= {neg, pos}

    named = {word for word in vocabulary if word in text}
    assert named, f"no scenario vocabulary appeared at all in: {text}"
    assert "stance_" not in text, "raw column names leaked into the narrative"


def test_narrate_off_by_default_and_costs_nothing():
    cfg = _cfg(n_ticks=5)
    assert all(state.summary is None for state in run_iter(cfg, seed=0))


def test_camp_language_is_gated_on_actual_bimodality():
    """`stance_clusters` splits at the median of the dominant component, so it
    returns two groups even for a unimodal cloud where the split is noise. A
    narrator that confidently describes polarization that is not there is the
    exact failure a normative study cannot afford.
    """
    lex = lexicon_for(_cfg())
    rng = np.random.default_rng(0)

    unimodal = rng.normal(0, 1, size=(800, 3))
    narrator = Narrator(lex=lex)
    summary = narrator.observe(
        t=0, s=np.ones(8), sigma=np.zeros((8, 3)), stance_u=unimodal
    )
    assert summary.bimodality <= BIMODAL_THRESHOLD
    assert not summary.has_camps
    assert "No clear camps" in describe_state(summary, lex)

    split = np.concatenate([rng.normal(-3, 0.3, (400, 3)), rng.normal(3, 0.3, (400, 3))])
    summary = Narrator(lex=lex).observe(t=0, s=np.ones(8), sigma=np.zeros((8, 3)), stance_u=split)
    assert summary.has_camps
    assert "bimodal" in describe_state(summary, lex)


def test_narrator_reports_no_camp_headcount():
    """A median split is exactly N/2 by construction, so printing the sizes
    presents an artifact as a measurement."""
    lex = lexicon_for(_cfg())
    rng = np.random.default_rng(0)
    split = np.concatenate([rng.normal(-3, 0.3, (700, 3)), rng.normal(3, 0.3, (100, 3))])
    summary = Narrator(lex=lex).observe(t=0, s=np.ones(8), sigma=np.zeros((8, 3)), stance_u=split)

    assert summary.camp_sizes == (400, 400), "median split should be even; if not, revisit the text"
    assert "400" not in describe_state(summary, lex)


def test_hardening_trend_tracks_the_direction_of_change():
    lex = lexicon_for(_cfg())
    narrator = Narrator(lex=lex, ewma=1.0)
    s = np.ones(8)
    stance = np.random.default_rng(0).normal(0, 1, (200, 3))

    narrator.observe(t=0, s=s, sigma=np.full((8, 3), 0.1), stance_u=stance)
    rising = narrator.observe(t=1, s=s, sigma=np.full((8, 3), 0.4), stance_u=stance)
    assert rising.hardening[0] > 0

    falling = narrator.observe(t=2, s=s, sigma=np.full((8, 3), 0.2), stance_u=stance)
    assert falling.hardening[0] < 0


def test_cross_cutting_sits_below_the_chance_baseline():
    """The narrator's cross-cutting number has to respond to something, or it
    is decoration. With two equal-sized camps, engagements pairing at random
    would cross about half the time; a homophilous population must sit below
    that.

    Deliberately NOT a lever contrast, but not because the levers do nothing.
    A `DiscourseSummary` describes ONE tick — roughly 130 engagements — which
    is far too few to separate conditions. Aggregated over a whole run the
    ranker moves this substantially: cross-camp share is 0.319 under
    `chronological` against 0.250 under `affinity` (n≈3200 engagements per
    run, 4 seeds), and 0.362 vs 0.288 measured over exposures.

    So the narrator's per-tick number is for reading the run as it happens,
    not for comparing conditions; that is what `outcomes.cross_cutting_exposure`
    over the persisted tables is for.
    """
    rates = np.array([_last_summary(_cfg(), seed).cross_cutting_share for seed in range(6)])
    assert np.all(np.isfinite(rates))
    assert rates.mean() < 0.5, f"no homophily at all in the engagement log: {rates}"
