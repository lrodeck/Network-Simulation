"""The semantic layer: names for what the model computes over.

The scenario format has always carried the vocabulary and only the LLM
renderer ever read it. These tests pin the naming surfaces, because the two
stance column conventions are easy to conflate and the failure is silent.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.data import load_scenario, packaged_scenarios, scenario_config
from discourse_lab.population import sample_population
from discourse_lab.semantics import Lexicon, lexicon_for


def test_packaged_scenario_loads_and_names_its_axes():
    assert "default" in packaged_scenarios()
    scenario = load_scenario("default")
    assert scenario.axis_names() == ("provision", "openness", "institutional trust")
    assert scenario.poles()[0] == ("market", "state")


def test_scenario_config_refuses_a_silent_dimensionality_change():
    """A scenario overrides `population.stance_dims` through
    `Config.stance_dims()`. That is not cosmetic — the homophily agreement
    effect against its matched null is +0.0049 (t=+2.46) at D=1 and -0.0001
    (t=-0.05) at D=3 — so the substitution has to be asked for.
    """
    base = dataclasses.replace(
        Config(), population=dataclasses.replace(Config().population, stance_dims=5)
    )
    with pytest.raises(ValueError, match="allow_dim_change"):
        scenario_config(base)

    assert scenario_config(base, allow_dim_change=True).stance_dims() == 3


def test_trait_column_matches_a_real_population_exactly():
    """The regression test for the space in "institutional trust".

    `population/traits.py` names stance columns `stance_{axis_name}` while
    `io/store.py` flattens them positionally to `stance_{d}`. Anything that
    assumes one convention breaks on the other.
    """
    cfg = scenario_config(
        dataclasses.replace(Config(), population=dataclasses.replace(Config().population, n_users=200))
    )
    pop = sample_population(cfg, np.random.default_rng(0))
    lex = lexicon_for(cfg)

    for d in range(lex.n_axes):
        assert lex.trait_column(d) in pop.trait_names, lex.trait_column(d)
    assert lex.trait_column(2) == "stance_institutional trust"

    # and resolving through the lexicon gives the same columns as the ad-hoc
    # prefix scan the rest of the codebase uses
    by_prefix = [i for i, n in enumerate(pop.trait_names) if n.startswith("stance_")]
    assert lex.stance_columns(pop.trait_names) == by_prefix


def test_stance_columns_resolves_the_positional_parquet_convention():
    lex = lexicon_for(scenario_config())
    parquet_names = ["id", "t", "stance_0", "stance_1", "stance_2"]
    assert lex.stance_columns(parquet_names) == [2, 3, 4]


def test_stance_columns_raises_rather_than_guessing():
    lex = lexicon_for(scenario_config())
    with pytest.raises(KeyError, match="stance axis"):
        lex.stance_columns(["id", "t", "stance_0"])


def test_generic_lexicon_when_no_scenario_is_attached():
    lex = Lexicon.from_config(Config())
    assert lex.trait_column(0) == "stance_0"
    assert lex.topic_label(3) == "topic 3"
    assert lex.pole_label(0, 1.0) == "+"


def test_named_topics_are_read_from_the_scenario():
    """`topic_names` was declared, validated, and silently dropped by
    `from_editor_json` — so it could never be populated."""
    lex = lexicon_for(scenario_config())
    assert lex.topic_label(0) == "immigration"
    assert lex.topic_label(99) == "topic 99"   # beyond the named set, still works


def test_config_rejects_more_topic_names_than_topics():
    from discourse_lab.config import ScenarioConfig

    scenario = ScenarioConfig(topic_names=tuple(f"t{i}" for i in range(20)))
    with pytest.raises(ValueError, match="n_topics"):
        dataclasses.replace(Config(), scenario=scenario)


def test_lexicon_is_memoised_on_the_config_hash():
    """`Config` holds plain dicts and is not hashable, so `lru_cache(cfg)`
    raises at the first call — the hash string is the key."""
    cfg = scenario_config()
    assert lexicon_for(cfg) is lexicon_for(dataclasses.replace(cfg))

    with pytest.raises(TypeError):
        hash(cfg)
