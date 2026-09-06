"""Step 4 verification (dev §6): degree distribution, clustering coefficient,
homophily; neighbourhoods measurably differ between generators.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from discourse_lab.config import Config
from discourse_lab.network import cached_graph, generate_graph
from discourse_lab.network.latent_space import _latent_coords
from discourse_lab.network.measures import degree_sequence, global_clustering, mean_neighbor_distance
from discourse_lab.network.reciprocity import add_reciprocity
from discourse_lab.population import sample_population


def _cfg(generator: str, n_users: int = 3000, **graph_overrides) -> Config:
    base = Config()
    return dataclasses.replace(
        base,
        population=dataclasses.replace(base.population, n_users=n_users),
        graph=dataclasses.replace(base.graph, generator=generator, **graph_overrides),
    )


def test_latent_space_hits_target_mean_degree_and_has_no_self_loops():
    cfg = _cfg("latent_space", mean_degree=30.0, mirror_p=0.0, long_tie_fraction=0.0)
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    g = generate_graph(cfg, pop, rng)

    deg = degree_sequence(g.csr)
    assert abs(deg.mean() - cfg.graph.mean_degree) < 6.0
    assert g.csr.diagonal().sum() == 0


def test_latent_space_is_more_clustered_than_configuration_null_model():
    rng_seed = 1
    latent_cfg = _cfg("latent_space", mean_degree=20.0)
    config_cfg = _cfg("configuration_model", mean_degree=20.0)

    rng = np.random.default_rng(rng_seed)
    pop = sample_population(latent_cfg, rng)
    g_latent = generate_graph(latent_cfg, pop, np.random.default_rng(2))
    g_config = generate_graph(config_cfg, pop, np.random.default_rng(2))

    c_latent = global_clustering(g_latent.csr)
    c_config = global_clustering(g_config.csr)
    assert c_latent > c_config  # homophily manufactures triangles the null model lacks


def test_latent_space_neighbourhoods_are_more_alike_than_chance():
    cfg = _cfg("latent_space", mean_degree=20.0)
    rng = np.random.default_rng(3)
    pop = sample_population(cfg, rng)
    g = generate_graph(cfg, pop, np.random.default_rng(4))
    coords = _latent_coords(pop)

    observed = mean_neighbor_distance(g.csr, coords)

    rng2 = np.random.default_rng(5)
    n = coords.shape[0]
    random_rows = rng2.integers(0, n, 20_000)
    random_cols = rng2.integers(0, n, 20_000)
    chance = np.linalg.norm(coords[random_rows] - coords[random_cols], axis=1).mean()

    assert observed < chance  # latent_space neighbours sit closer than random pairs


def test_sbm_neighbourhoods_differ_from_configuration_model():
    n_users = 3000
    sbm_cfg = _cfg("sbm", n_users=n_users, mean_degree=20.0, sbm_homophily=0.9)
    config_cfg = _cfg("configuration_model", n_users=n_users, mean_degree=20.0)

    rng = np.random.default_rng(6)
    pop = sample_population(sbm_cfg, rng)
    g_sbm = generate_graph(sbm_cfg, pop, np.random.default_rng(7))
    g_config = generate_graph(config_cfg, pop, np.random.default_rng(7))

    def same_archetype_share(g):
        coo = g.csr.tocoo()
        labels = pop.archetype_labels
        return (labels[coo.row] == labels[coo.col]).mean()

    share_sbm = same_archetype_share(g_sbm)
    share_config = same_archetype_share(g_config)
    assert share_sbm > share_config  # SBM imposes block structure the null model lacks


def test_barabasi_albert_degree_is_heavy_tailed():
    cfg = _cfg("barabasi_albert", n_users=1500, mean_degree=10.0, mirror_p=0.0)
    rng = np.random.default_rng(8)
    pop = sample_population(cfg, rng)
    g = generate_graph(cfg, pop, rng)

    deg = degree_sequence(g.csr + g.csr.T)
    assert deg.max() > 6 * deg.mean()  # a few hubs dominate, unlike a Poisson degree sequence


def test_reciprocity_pass_hits_target_ratio():
    n = 4000
    rng = np.random.default_rng(9)
    density = 0.01
    from scipy import sparse

    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    G = sparse.csr_matrix(mask.astype(np.int8))

    r = 0.3
    G2 = add_reciprocity(G, r, np.random.default_rng(10))

    coo = G.tocoo()
    has_reverse_before = 0
    reverse_lookup = set(zip(coo.row.tolist(), coo.col.tolist()))
    coo2 = G2.tocoo()
    pairs2 = set(zip(coo2.row.tolist(), coo2.col.tolist()))

    original_pairs = set(zip(coo.row.tolist(), coo.col.tolist()))
    one_directional = [(u, v) for (u, v) in original_pairs if (v, u) not in original_pairs]
    newly_reciprocated = sum(1 for (u, v) in one_directional if (v, u) in pairs2)

    observed_ratio = newly_reciprocated / len(one_directional)
    assert abs(observed_ratio - r) < 0.05


def test_graph_artifact_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = _cfg("configuration_model", n_users=500)
    rng = np.random.default_rng(11)
    pop = sample_population(cfg, rng)

    g1 = cached_graph(cfg, seed=0, pop=pop, rng=rng)
    g2 = cached_graph(cfg, seed=0, pop=pop, rng=np.random.default_rng(999))

    np.testing.assert_array_equal(g1.csr.toarray(), g2.csr.toarray())
