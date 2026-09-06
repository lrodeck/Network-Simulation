"""The per-user view: who they are, who they follow, and what they were shown.

The load-bearing claim is `feed_composition`'s baseline. It must be sensitive
to the ranker in the right direction, or `bubble` is a number that measures the
population's stance spread and nothing else.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.data import scenario_config
from discourse_lab.network import cached_graph
from discourse_lab.population import cached_population
from discourse_lab.runner import cached_run, load_run
from discourse_lab.semantics import lexicon_for
from discourse_lab.users import audience_summary, feed_composition, user_table

PERSIST = ("posts", "engagements", "exposures", "traits")


def _world(ranker: str, seed: int = 0, n_users: int = 400, n_ticks: int = 20):
    cfg = scenario_config(dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=n_users, n_topics=8),
        dynamics=dataclasses.replace(
            Config().dynamics, n_ticks=n_ticks, ranker=ranker, drift="none",
            exposure_sample_rate=0.10),
    ))
    rng = np.random.default_rng(seed)
    pop = cached_population(cfg, seed=seed, rng=rng)
    graph = cached_graph(cfg, seed=seed, pop=pop, rng=rng)
    cached_run(cfg, seed=seed, persist=PERSIST)
    return cfg, pop, graph, load_run(cfg, seed=seed), lexicon_for(cfg)


def test_user_table_names_stance_through_the_lexicon():
    cfg, pop, graph, handle, lex = _world("affinity")
    table = user_table(handle, pop, graph, lex)

    assert table.height == cfg.population.n_users
    for d in range(lex.n_axes):
        assert lex.trait_column(d) in table.columns
    assert "stance_0" not in table.columns, "positional column leaked into the user view"
    assert set(table["archetype"].unique()) <= set(pop.archetype_names)
    # follows counts rows of the CSR, followers counts columns — swapping them is
    # silent and wrong in exactly the direction that flatters a graph generator
    assert table["follows"].sum() == table["followers"].sum() == graph.csr.nnz


def test_feed_is_closer_to_the_viewer_than_the_world_is():
    """The filter-bubble effect, measured per user against same-tick posts."""
    _, pop, _, handle, lex = _world("affinity")
    feeds = feed_composition(handle, pop, lex)

    assert feeds.height > 0
    assert feeds["bubble"].mean() < 0
    assert np.isfinite(feeds["world_dist"].to_numpy()).all()


def test_affinity_ranking_narrows_the_feed_more_than_chronological():
    """Sign test over seeds, not a single run: this is the whole reason the
    baseline exists, so it has to move with the lever."""
    gaps = []
    for seed in range(3):
        _, pop_a, _, handle_a, lex_a = _world("affinity", seed=seed)
        _, pop_c, _, handle_c, lex_c = _world("chronological", seed=seed)
        gaps.append((feed_composition(handle_a, pop_a, lex_a)["bubble"].mean(),
                     feed_composition(handle_c, pop_c, lex_c)["bubble"].mean()))

    assert all(a < c for a, c in gaps), gaps


def test_affinity_ranking_narrows_topic_variety_too():
    _, pop_a, _, handle_a, lex_a = _world("affinity")
    _, pop_c, _, handle_c, lex_c = _world("chronological")
    affinity = feed_composition(handle_a, pop_a, lex_a)
    chrono = feed_composition(handle_c, pop_c, lex_c)

    # the world baseline is the same run-to-run, so a drop in feed entropy that
    # is not matched in world entropy is the ranker's doing
    assert affinity["feed_topic_entropy"].mean() < chrono["feed_topic_entropy"].mean()
    assert affinity["feed_topic_entropy"].mean() < affinity["world_topic_entropy"].mean()


def test_audience_summary_drops_thin_feeds_and_says_so():
    _, pop, graph, handle, lex = _world("affinity")
    users, feeds = user_table(handle, pop, graph, lex), feed_composition(handle, pop, lex)

    loose = audience_summary(users, feeds, min_seen=1)
    strict = audience_summary(users, feeds, min_seen=10_000)
    assert loose["n"].sum() > 0
    assert strict.height == 0 or strict["n"].sum() == 0


def test_feed_composition_requires_the_tables_it_reads():
    cfg = scenario_config(dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=200, n_topics=8),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=5, drift="none"),
    ))
    rng = np.random.default_rng(0)
    pop = cached_population(cfg, seed=0, rng=rng)
    cached_run(cfg, seed=0, persist=("posts",))
    with pytest.raises(ValueError, match="exposures"):
        feed_composition(load_run(cfg, seed=0), pop, lexicon_for(cfg))
