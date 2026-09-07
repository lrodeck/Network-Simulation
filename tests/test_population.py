"""Step 2 verification (dev §6): marginals + rank correlations match spec,
drawn scenario curves reproduce in the sample.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy import stats

from discourse_lab.config import Config, ScenarioConfig
from discourse_lab.population import cached_population, sample_population
from discourse_lab.population.copula import nearest_psd_correlation, sparse_pairs_to_matrix
from discourse_lab.population.marginals import empirical_from_editor

RNG = np.random.default_rng(0)


def _big_population():
    cfg = dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=20_000),
    )
    return cfg, sample_population(cfg, np.random.default_rng(1))


def test_marginals_match_spec_families():
    cfg, pop = _big_population()
    names = pop.trait_names

    activity = pop.X_used[:, names.index("activity")]
    # Lognormal(mu=0, sigma=activity_sigma): mean of log(activity) ~ 0, std ~ sigma
    logs = np.log(activity)
    assert abs(logs.mean()) < 0.1
    assert abs(logs.std() - cfg.population.activity_sigma) < 0.15

    prominence = pop.X_used[:, names.index("prominence")]
    # Pareto(alpha): P(X > x) = x^-alpha for x >= 1 -> log-log slope is -alpha
    assert prominence.min() >= 1.0
    tail = np.sort(prominence)[::-1][: len(prominence) // 20]
    assert tail.mean() > np.median(prominence)  # heavy right tail

    contrarianism = pop.X_used[:, names.index("contrarianism")]
    assert 0.0 < contrarianism.min() and contrarianism.max() < 1.0
    empirical = np.sort(contrarianism)
    theoretical_cdf = stats.beta(2, 5).cdf(empirical)
    ranks = (np.arange(len(empirical)) + 1) / len(empirical)
    max_gap = np.max(np.abs(ranks - theoretical_cdf))
    assert max_gap < 0.05

    plasticity = pop.X_used[:, names.index("plasticity")]
    conviction = pop.X_used[:, names.index("conviction")]
    assert plasticity.mean() < 0.5  # Beta(2,8): most people barely move
    assert conviction.mean() > 0.5  # Beta(5,2): stance stickier than style

    personality = pop.X_used[:, names.index("openness")]
    assert abs(personality.mean()) < 0.1
    assert abs(personality.std() - 1.0) < 0.1


def test_rank_correlations_match_configured_pairs():
    cfg = dataclasses.replace(
        Config(),
        population=dataclasses.replace(
            Config().population,
            n_users=15_000,
            correlation_pairs=(("activity", "prominence", 0.6),),
        ),
    )
    pop = sample_population(cfg, np.random.default_rng(2))
    names = pop.trait_names
    a = pop.X_used[:, names.index("activity")]
    p = pop.X_used[:, names.index("prominence")]
    rho, _ = stats.spearmanr(a, p)
    assert rho > 0.3  # monotonic transform preserves the sign/rough magnitude


def test_correlation_completion_is_psd_and_respects_sparse_pairs():
    names = ["a", "b", "c"]
    pairs = (("a", "b", 0.9), ("b", "c", 0.9), ("a", "c", -0.9))  # inconsistent triple
    raw = sparse_pairs_to_matrix(names, pairs)
    fixed = nearest_psd_correlation(raw)

    eigvals = np.linalg.eigvalsh(fixed)
    assert eigvals.min() > -1e-8
    assert np.allclose(np.diag(fixed), 1.0, atol=1e-6)


def test_scenario_curve_reproduces_in_drawn_sample():
    path = Path(__file__).resolve().parents[1] / "discourse_lab/data/scenarios/default.json"
    data = json.loads(path.read_text())
    scenario = ScenarioConfig.from_editor_json(data, name="default")
    cfg = dataclasses.replace(
        Config(),
        scenario=scenario,
        population=dataclasses.replace(Config().population, n_users=20_000),
    )
    pop = sample_population(cfg, np.random.default_rng(3))

    axis = scenario.stance_axes[0]
    m = axis["marginal"]
    lo, hi = m["support"]
    density = np.asarray(m["density"])
    bins = m["bins"]

    marginal = empirical_from_editor(bins=bins, support=(lo, hi), density=list(density))
    draws = marginal.icdf(np.random.default_rng(4).uniform(size=100_000))

    stance = pop.X_used[:, pop.trait_names.index(f"stance_{axis['name']}")]

    edges = np.linspace(lo, hi, bins + 1)
    hist_target, _ = np.histogram(draws, bins=edges, density=True)
    hist_sample, _ = np.histogram(stance, bins=edges, density=True)

    # coarse-bin correlation between the empirical curve and the histogram of
    # the actually-drawn population values
    coarse = 16
    step = bins // coarse
    target_coarse = hist_target[: coarse * step].reshape(coarse, step).mean(axis=1)
    sample_coarse = hist_sample[: coarse * step].reshape(coarse, step).mean(axis=1)
    corr = np.corrcoef(target_coarse, sample_coarse)[0, 1]
    assert corr > 0.9


def test_population_artifact_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = Config()
    rng = np.random.default_rng(5)

    pop1 = cached_population(cfg, seed=0, rng=rng)
    pop2 = cached_population(cfg, seed=0, rng=np.random.default_rng(999))  # ignored: cache hit

    np.testing.assert_array_equal(pop1.X_used, pop2.X_used)
    assert pop1.trait_names == pop2.trait_names


def test_requested_correlation_adds_to_what_archetypes_already_induce():
    """`correlation_pairs` reads as "set this correlation" and behaves as "add
    to whatever the mixture produces". Both mechanisms are legitimate; the trap
    is that neither knows about the other.

    Measured at N=20000 for activity x reply_prop: archetypes off gives -0.001
    with no pairs and +0.299 when 0.30 is asked for; archetypes on gives +0.303
    with no pairs and +0.493 when the same 0.30 is asked for.
    """
    import dataclasses

    from discourse_lab.config import Config

    def realised(pairs=(), weights=()):
        cfg = dataclasses.replace(
            Config(),
            population=dataclasses.replace(
                Config().population, n_users=6000,
                correlation_pairs=pairs, archetype_weights=weights,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pop = sample_population(cfg, np.random.default_rng(0))
        corr = np.corrcoef(pop.X_stored.T)
        i, j = pop.trait_names.index("activity"), pop.trait_names.index("reply_prop")
        return corr[i, j]

    flat = (("lurker", 1.0),)      # one archetype carrying all weight = no offsets applied
    assert abs(realised(weights=flat)) < 0.05
    assert realised(pairs=(("activity", "reply_prop", 0.30),), weights=flat) == pytest.approx(
        0.30, abs=0.05
    ), "without archetypes you should get exactly what you ask for"

    induced = realised()
    assert induced > 0.2, "the lurker archetype should correlate activity and reply_prop"
    both = realised(pairs=(("activity", "reply_prop", 0.30),))
    assert both > induced + 0.1, "the two mechanisms should compose, not reconcile"


def test_composed_correlation_warns():
    """Asking for 0.30 and silently getting 0.49 is the kind of thing that
    survives into a paper as "we set trait correlation to 0.3"."""
    import dataclasses

    from discourse_lab.config import Config

    cfg = dataclasses.replace(
        Config(),
        population=dataclasses.replace(
            Config().population, n_users=200,
            correlation_pairs=(("activity", "reply_prop", 0.30),),
        ),
    )
    with pytest.warns(UserWarning, match="lurker"):
        sample_population(cfg, np.random.default_rng(0))


def test_no_warning_when_the_pair_is_untouched_by_any_archetype():
    """The warning has to be specific or it becomes noise people filter out."""
    import dataclasses

    from discourse_lab.config import Config

    cfg = dataclasses.replace(
        Config(),
        population=dataclasses.replace(
            Config().population, n_users=200,
            correlation_pairs=(("openness", "neuroticism", 0.30),),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample_population(cfg, np.random.default_rng(0))
