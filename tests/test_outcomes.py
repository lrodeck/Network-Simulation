"""Normative outcomes — the dependent variables of a democratic-discourse study.

These are constructs, not just numbers, so the tests check that each responds
to the mechanism it claims to measure and that the reporting cannot be read the
wrong way round.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.data import scenario_config
from discourse_lab.outcomes import (
    cross_cutting_exposure,
    hostility_given_contact,
    normative_outcomes,
    outcome_names,
    voice_inequality,
)
from discourse_lab.population import cached_population
from discourse_lab.runner import cached_run, load_run, phase_rngs
from discourse_lab.semantics import lexicon_for

PERSIST = ("posts", "engagements", "exposures", "traits")


def _cfg(n_users=600, n_ticks=20, sample=0.10, **dyn):
    return scenario_config(
        dataclasses.replace(
            Config(),
            population=dataclasses.replace(Config().population, n_users=n_users, stance_dims=3),
            dynamics=dataclasses.replace(
                Config().dynamics, n_ticks=n_ticks, drift="none",
                exposure_sample_rate=sample, **dyn
            ),
        )
    )


def _run(cfg, seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cached_run(cfg, seed=seed, persist=PERSIST)
    handle = load_run(cfg, seed=seed)
    pop = cached_population(cfg, seed, phase_rngs(seed)["population"])
    return handle, pop, lexicon_for(cfg)


def test_a_bubble_maximal_ranker_lowers_cross_cutting_exposure(tmp_path, monkeypatch):
    """spec §2.5 calls `affinity` "filter bubble maximal". If the construct
    does not fall under it, the construct is decoration.

    Both readings must move together — the coarse camp split and the continuous
    stance distance are different operationalisations of one idea, and if they
    disagree the idea is not being measured.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    chrono = cross_cutting_exposure(*_run(_cfg(ranker="chronological")))
    affinity = cross_cutting_exposure(*_run(_cfg(ranker="affinity")))

    assert affinity["camp_share"] < chrono["camp_share"]
    assert affinity["stance_share"] < chrono["stance_share"]
    assert 0.0 < affinity["camp_share"] < 0.5, "below the two-camp chance baseline"


def test_delta_scales_with_stance_dimensionality(tmp_path, monkeypatch):
    """Stance distances grow like sqrt(D), so a fixed threshold means different
    things at different D. An earlier default of 0.5 * per-axis sd put delta at
    0.25 against typical distances near 0.87 and called 94% of everything
    cross-cutting — a number that moved with nothing.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    result = cross_cutting_exposure(*_run(_cfg()))
    assert 0.05 < result["stance_share"] < 0.95, result
    assert result["delta"] > 0.3, "delta collapsed toward the per-axis scale again"


def test_algorithmic_share_isolates_what_a_recommender_controls(tmp_path, monkeypatch):
    """Injected exposures are the part the platform chooses. Measured, they are
    far more cross-cutting than the feed they land in while barely moving the
    aggregate — the mechanism works per item and is drowned out by follower
    fanout.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    result = cross_cutting_exposure(*_run(_cfg(ranker="affinity", inject_k=10)))

    assert np.isfinite(result["algorithmic_share"])
    assert result["algorithmic_share"] > result["camp_share"]

    # and with no injection there is nothing to isolate, so it must say so
    # rather than quietly reporting the subscription feed
    none = cross_cutting_exposure(*_run(_cfg(ranker="affinity", inject_k=0)))
    assert np.isnan(none["algorithmic_share"])


def test_hostility_is_reported_with_its_contact_rate(tmp_path, monkeypatch):
    """A platform that eliminates cross-camp contact scores a perfect zero on
    hostility while being the opposite of the intervention's goal. The two
    numbers are only interpretable together, so the API must return both.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    result = hostility_given_contact(*_run(_cfg()))
    assert set(result) >= {"contact_rate", "hostility", "n"}
    assert np.isfinite(result["contact_rate"]) and np.isfinite(result["hostility"])


def test_voice_inequality_reports_the_minority_camp_separately(tmp_path, monkeypatch):
    """An aggregate Gini can look healthy while one camp is inaudible."""
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    result = voice_inequality(*_run(_cfg()))
    assert 0.0 <= result["minority_camp_attention_share"] <= 1.0
    assert 0.0 <= result["posting_gini"] <= 1.0
    assert 0.0 <= result["attention_gini"] <= 1.0


def test_normative_outcomes_degrades_when_tables_are_missing(tmp_path, monkeypatch):
    """Same contract as `stylized_facts_from_run`: omit what cannot be
    computed rather than fail or report a wrong number."""
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = _cfg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cached_run(cfg, seed=1, persist=("posts",))     # no exposures, no engagements
    handle = load_run(cfg, seed=1)
    pop = cached_population(cfg, 1, phase_rngs(1)["population"])

    out = normative_outcomes(handle, pop=pop, lex=lexicon_for(cfg))
    assert any(k.startswith("voice_inequality") for k in out)
    assert not any(k.startswith("cross_cutting_exposure") for k in out)
    assert not any(k.startswith("hostility_given_contact") for k in out)


def test_every_outcome_is_registered_and_reachable_by_name():
    assert set(outcome_names()) == {
        "cross_cutting_exposure", "epistemic_alignment",
        "hostility_given_contact", "voice_inequality",
    }
