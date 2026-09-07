"""The attention budget and the position decay are two caps on the same thing,
and the softer one binds first.

This is not a hypothetical. A 10-seed intervention sweep reported
`attention_budget` as inert on all 13 normative outcome columns, because it was
swept at 15/30/60 — a range in which the cap removes 10%, 1% and 0% of what the
position decay has already let through. The test exists so the next person to
pick a sweep range finds the interaction pinned rather than rediscovering it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from discourse_lab.config import Config
from discourse_lab.exposure.attention import select_exposures
from discourse_lab.exposure.inbox import CandidatePairs


def _binding_fraction(budget: float, tau: float, n_candidates: int = 200,
                      trials: int = 400) -> float:
    """Share of position-decay survivors that the budget additionally removes."""
    rng = np.random.default_rng(0)
    kept_both = kept_decay = 0
    for _ in range(trials):
        pairs = CandidatePairs(
            post_idx=np.arange(n_candidates),
            user_id=np.zeros(n_candidates, dtype=np.int64),
            is_follower=np.ones(n_candidates, dtype=bool),
        )
        scores = np.linspace(1.0, 0.0, n_candidates)
        activity = np.ones(1)
        kept_both += len(select_exposures(pairs, scores, activity, budget, tau, rng))
        kept_decay += len(select_exposures(pairs, scores, activity, 1e9, tau, rng))
    return 1.0 - kept_both / kept_decay


def test_position_decay_passes_about_tau_items():
    """The ceiling every budget has to get under to matter."""
    rng = np.random.default_rng(0)
    n = 400
    pairs = CandidatePairs(post_idx=np.arange(n), user_id=np.zeros(n, dtype=np.int64),
                           is_follower=np.ones(n, dtype=bool))
    seen = np.mean([
        len(select_exposures(pairs, np.linspace(1, 0, n), np.ones(1), 1e9, 6.0, rng))
        for _ in range(200)])
    assert 5.0 < seen < 8.0, seen


def test_the_shipped_default_budget_barely_binds():
    """Not an assertion that this is wrong — an assertion that it is known."""
    assert _binding_fraction(Config().dynamics.attention_budget,
                             Config().dynamics.tau_position) < 0.05


def test_a_low_budget_actually_rations_attention():
    assert _binding_fraction(3.0, 6.0) > 0.4


def test_budget_is_monotone_in_its_binding_range():
    fractions = [_binding_fraction(b, 6.0) for b in (3.0, 10.0, 30.0)]
    assert fractions[0] > fractions[1] > fractions[2]


def test_default_lever_range_spans_the_binding_threshold():
    """The sweep range must contain values on both sides of "binds at all", or
    the study measures three copies of the same platform."""
    from discourse_lab.experiments.intervention import DEFAULT_LEVERS

    values = DEFAULT_LEVERS["dynamics.attention_budget"]
    fractions = [_binding_fraction(float(b), Config().dynamics.tau_position) for b in values]
    assert min(fractions) < 0.05 and max(fractions) > 0.4, dict(zip(values, fractions))


def test_tau_position_interacts_with_the_ranker():
    """The main-effects sweep reports `tau_position` as inert. It is inert
    *at the base ranker*, and that is not the same claim.

    Under `chronological` the rank order is recency, independent of stance, so
    truncating the feed removes a random slice. Under `affinity` the rank order
    is stance order, so truncating removes precisely the disagreement. Kept
    small and sign-only — the magnitudes live in the module docstring.
    """
    import warnings

    from discourse_lab.data import scenario_config
    from discourse_lab.outcomes import cross_cutting_exposure
    from discourse_lab.population import cached_population
    from discourse_lab.runner import cached_run, load_run, phase_rngs
    from discourse_lab.semantics import lexicon_for

    def camp_share(ranker: str, tau: float, seed: int) -> float:
        cfg = scenario_config(dataclasses.replace(
            Config(),
            population=dataclasses.replace(Config().population, n_users=500, stance_dims=3),
            dynamics=dataclasses.replace(
                Config().dynamics, n_ticks=20, ranker=ranker, tau_position=tau,
                drift="none", exposure_sample_rate=0.10),
        ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cached_run(cfg, seed=seed, persist=("posts", "engagements", "exposures", "traits"))
        pop = cached_population(cfg, seed, phase_rngs(seed)["population"])
        return cross_cutting_exposure(load_run(cfg, seed=seed), pop, lexicon_for(cfg))["camp_share"]

    seeds = range(3)
    chrono = np.mean([camp_share("chronological", 15.0, s) - camp_share("chronological", 2.0, s)
                      for s in seeds])
    affinity = np.mean([camp_share("affinity", 15.0, s) - camp_share("affinity", 2.0, s)
                        for s in seeds])

    assert affinity > 0.03, affinity
    assert affinity > 5 * abs(chrono), (affinity, chrono)
