"""Conformance tests for the change-spec mechanisms (C1-C10).

House style (tests/test_spec_conformance.py): each test observes the
mechanism's *effect*, never its definition — a mechanism that is wired but
does nothing must fail here, not pass.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.runner import PHASES, phase_rngs, run_iter


def _cfg(n_users: int = 400, n_ticks: int = 12, **dyn):
    return dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=n_users, **dyn.pop("pop", {})),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=n_ticks, **dyn),
    )


# --------------------------------------------------------------------------
# Wave 0: structural invariants
# --------------------------------------------------------------------------


def test_phase_prefix_is_frozen():
    """Change spec §0.2: PHASES is append-only. The first nine phases are the
    pre-change-spec set; selection/rewire/affect were appended in one commit,
    in that order. Inserting rather than appending would shift every
    subsequent stream and silently change every existing seed's results."""
    assert PHASES[:9] == ("population", "graph", "timing", "generation",
                          "exposure", "reaction", "perception", "cascade", "drift")
    assert PHASES[9:] == ("selection", "rewire", "affect")


def test_schema_version_is_recorded_in_the_run_meta():
    """§0.1: future cache forks should be legible, not mysterious."""
    import tempfile
    from pathlib import Path

    from discourse_lab.config import Config
    from discourse_lab.io.workspace import workspace

    old = dict(__import__("os").environ)
    import os

    os.environ["DLAB_HOME"] = tempfile.mkdtemp()
    try:
        cfg = _cfg(n_users=200, n_ticks=2)
        from discourse_lab.io.store import RunWriter

        writer = RunWriter(Path(workspace() / "meta-test"), cfg, 0)
        import json

        meta = json.loads((writer.path / "meta.json").read_text(encoding="utf-8"))
        writer.close()
        assert meta["schema_version"] == cfg.schema_version
    finally:
        __import__("os").environ.clear()
        __import__("os").environ.update(old)


def test_kernel_learning_never_invalidates_the_population_artifact():
    """C3b's state-location decision exists so a kernel swap does not fork the
    population cache. `kernel_learning` lives in DynamicsConfig; the
    population key is structural over (population, scenario) only."""
    from discourse_lab.io.artifacts import artifact_paths

    base = _cfg()
    learned = _cfg(kernel_learning="group_gain", kernel_learning_rule="conformity")
    assert (artifact_paths(base)["population"] == artifact_paths(learned)["population"])
    # ...while an actual population change must fork it
    assert (artifact_paths(base)["population"] != artifact_paths(_cfg(pop={"affect": True}))["population"])


# --------------------------------------------------------------------------
# Wave 1: C4 quality backdoor, C5 out-group feature, C9 gate
# --------------------------------------------------------------------------


def test_c4_quality_coupling_zero_breaks_the_trait_backdoor():
    """At coupling 0 quality must be author-trait-independent: the quality row
    of A is zero, so author traits cannot reach it. At 1 the backdoor the
    change spec documents is present (and the metric must be read as a
    difference from null)."""
    from discourse_lab.dynamics.expression import ExpressionMap
    from discourse_lab.dynamics.posts import generate_posts

    cfg = _cfg(n_users=600)
    rng = np.random.default_rng(0)
    from discourse_lab.population import sample_population

    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()

    quality_cols = []
    for coupling in (0.0, 1.0):
        expr = ExpressionMap.build(pop.trait_names, K, quality_trait_coupling=coupling)
        authors = rng.integers(0, 600, 400)
        posts = generate_posts(authors, pop, expr, np.zeros(K), np.zeros((K, D)), 0.3, rng)
        from scipy import stats

        rho, _ = stats.spearmanr(posts.quality, pop.X_used[authors, pop.trait_names.index("conscientiousness")])
        quality_cols.append(rho)

    assert abs(quality_cols[0]) < 0.15, (
        f"coupling 0 still correlates quality with author traits (rho={quality_cols[0]:.2f})"
    )
    assert quality_cols[1] != quality_cols[0]


def test_c4_quality_attention_lift_requires_a_null_run():
    from discourse_lab.config import Config
    from discourse_lab.outcomes import quality_attention_lift

    class _Fake:
        def posts(self):
            raise AssertionError("lift read a run without a null to subtract")

    with pytest.raises(Exception):
        quality_attention_lift(_Fake(), None)


def test_c4_lift_rejects_an_unmatched_null():
    """C1/C2 made the naive null dangerous: differencing against a bare
    default-config null also differences out selection and affect, not just
    the kernel. The lift asserts its two runs' configs differ in
    `dynamics.kernel` alone and refuses anything else — cheap, and the kind
    of thing that silently produces a plausible wrong number."""
    import json
    import os
    import tempfile
    from pathlib import Path

    from discourse_lab.io.store import RunHandle, RunWriter
    from discourse_lab.outcomes import quality_attention_lift

    old = dict(os.environ)
    os.environ["DLAB_HOME"] = tempfile.mkdtemp()
    try:
        cfg_model = _cfg(n_users=200, n_ticks=2, kernel="bandwagon",
                         selection="homophilous")
        # the trap: a "null" that also switched selection off — plausible
        # looking, and it differences out the selection mechanism
        cfg_bad_null = _cfg(n_users=200, n_ticks=2, kernel="null")

        def _fake_run(cfg, name):
            p = Path(tempfile.mkdtemp()) / name
            w = RunWriter(p, cfg, 0)
            w.close()
            return RunHandle(p)

        model = _fake_run(cfg_model, "m")
        bad_null = _fake_run(cfg_bad_null, "n")
        with pytest.raises(ValueError, match="dynamics.kernel alone"):
            quality_attention_lift(model, bad_null)

        # the honest pairing clears the config check (it then fails later
        # only because these fake runs have no posts — which proves the
        # assertion is what stopped the first call)
        cfg_good_null = dataclasses.replace(
            cfg_model, dynamics=dataclasses.replace(cfg_model.dynamics, kernel="null"))
        good_null = _fake_run(cfg_good_null, "g")
        with pytest.raises(FileNotFoundError):
            quality_attention_lift(model, good_null)
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_theta_scale_is_a_population_level_group_lever():
    """C3b's group declarations, second consumer: `theta_scale` scales a
    whole named group at the population level — the lever D3's social_proof
    ladder needs. Swapping whole kernels cannot express a ladder (epistemic
    has no social_proof to weaken; bandwagon has no quality to crowd out),
    so this must not degrade into kernel swaps. Two properties pinned:
    scale=1 is exactly the unscaled kernel, and overrides land before the
    group scale multiplies them."""
    from discourse_lab.exposure.kernel import kernel_with_scales, named_kernel

    base = {(a, f): w for a, f, w in named_kernel("bandwagon")}
    scaled = {(a, f): w for a, f, w in kernel_with_scales(
        "bandwagon", scales=(("social_proof", 2.0),))}
    for key, w in base.items():
        action, feature = key
        in_group = feature in ("social_proof", "prominence", "tie_strength")
        expected = w * 2.0 if in_group else w
        assert scaled[key] == pytest.approx(expected), key

    # override then scale: the override sets the coefficient, the group
    # scale multiplies whatever it now is
    both = {(a, f): w for a, f, w in kernel_with_scales(
        "bandwagon",
        overrides=(("like", "social_proof", 0.5),),
        scales=(("social_proof", 2.0),))}
    assert both[("like", "social_proof")] == pytest.approx(1.0)
    assert both[("like", "intercept")] == base[("like", "intercept")]


def test_c5_outgroup_feature_is_unconditional_and_camp_signed():
    """The `outgroup` feature fires on camp difference alone — the Rathje et
    al. (2021) effect is general, not a contrarian-minority phenomenon. The
    kernel authoring must agree: `outrage` carries it positively,
    `homophily` negatively."""
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.exposure.kernel import compute_features, named_kernel

    cfg = _cfg(n_users=200)
    rng = np.random.default_rng(0)
    from discourse_lab.population import sample_population

    pop = sample_population(cfg, rng)
    camps = (np.arange(200) % 2).astype(np.int64)  # two artificial camps

    posts = _tiny_posts(cfg, rng)
    exposures = Exposures(
        post_idx=np.zeros(4, dtype=int),
        user_id=np.array([0, 1, 0, 1]),
        rank=np.zeros(4, dtype=int),
        is_follower=np.ones(4, dtype=bool),
    )
    feats = compute_features(exposures, posts, pop, exposures.is_follower, t_current=0, camps=camps)

    author_camp = camps[posts.author[0]]
    expected = (np.array([0, 1, 0, 1]) != author_camp).astype(float)
    np.testing.assert_array_equal(feats["outgroup"], expected)

    def _weight(kernel, action, feature):
        return {(a, f): w for a, f, w in named_kernel(kernel)}.get((action, feature), 0.0)

    assert _weight("outrage", "reply", "outgroup") > 0
    assert _weight("homophily", "like", "outgroup") < 0


def test_c9_attention_gini_and_reciprocity_hold_simultaneously():
    """Change spec C9's gate, post-C1-C10: attention Gini in [0.8, 0.95] AND
    reciprocity in [0.2, 0.4] **simultaneously**, across 20 seeds.

    C9 changed the graph generator and C1/C2/C3 changed the tick; every
    stylized-fact number previously calibrated was fitted under the old
    ones. This gate runs the calibrated combination under the CURRENT code
    at the calibration convention (`drift="none"`, like every other
    stylized-fact test — feed dynamics isolated from trait feedback), with
    the C1-C10 mechanisms at their identity settings: affect/learning/
    selection/rewiring off, quality_trait_coupling 0, C5's outgroup feature
    live as a feature.

    Known interaction, deliberately NOT tuned away: with the shipped
    `drift="full"` the Gini overshoots the band (~0.97 at this scale) —
    C3a's behavior reinforcement compounds attention concentration under
    bandwagon, which is the mechanism working exactly as Brady et al.
    describe and landing outside §5.1's range. Recorded in FINDINGS.md;
    experiments quoting stylized-anchored results WITH drift on must re-run
    this gate at their settings.

    Reduced scale (N=1200, 60 ticks) keeps the 20-seed gate inside a few
    minutes; the full eight-fact table at production scale is
    `stylized_facts_from_run` over the N=1e4 x 500 config and should be re-
    run before quoting any experiment result. Seeds overridable via
    DLAB_GATE_SEEDS for smoke runs."""
    import os

    from discourse_lab.analysis import set_param
    from discourse_lab.metrics import stylized_facts_from_run
    from discourse_lab.runner import cached_run, load_run

    seeds = int(os.environ.get("DLAB_GATE_SEEDS", "20"))
    base = _cfg(n_users=1200, n_ticks=60, drift="none")
    base = set_param(base, "dynamics.ranker", "engagement_optimized")
    base = set_param(base, "dynamics.kernel", "bandwagon")
    base = set_param(base, "graph.generator", "latent_pa")
    # Recalibrated post-C1-C10. Measured at 4 seeds each, then verified at
    # 20 (the 4-seed ranges understate the right tail — seed 9 at scale 0.75
    # hit 0.956, which is why the gate is 20 seeds and not 4):
    #   mirror_p 0.02, scale 1.00 -> Gini 0.939-0.964+ (over the top),
    #                              reciprocity 0.211-0.220 (skimming the floor)
    #   mirror_p 0.05, scale 0.75 -> Gini up to 0.956 at 20 seeds
    #   mirror_p 0.05, scale 0.60 -> Gini 0.82-0.93, reciprocity 0.252-0.268
    # The dials are the legitimate ones for each row: mirror_p is the graph's
    # reciprocity input (the PA overlay depresses it), theta_scale's
    # social_proof group is the popularity-force strength — the population-
    # level lever C3b's groups exist for. Calibration, not gate-fitting:
    # each dial moves its own row and both sit mid-band with margin.
    base = set_param(base, "graph.mirror_p", 0.05)
    base = set_param(base, "dynamics.theta_scale", (("social_proof", 0.75),))

    from discourse_lab.network import cached_graph
    from discourse_lab.population import cached_population

    for seed in range(seeds):
        cached_run(base, seed, persist=("posts", "engagements"))
        handle = load_run(base, seed)
        rngs = phase_rngs(seed)
        graph = cached_graph(base, seed, cached_population(base, seed, rngs["population"]), rngs["graph"])
        report = stylized_facts_from_run(handle, graph=graph, pop=None)
        for row in ("attention_gini", "reciprocity"):
            entry = report.get(row, {})
            assert entry.get("in_range") is True, (
                f"seed {seed}: {row} = {entry.get('value')} outside {entry.get('target')} — "
                f"the C9 gate half-succeeded (full report: "
                f"{ {k: v.get('value') for k, v in report.items()} })"
            )


# --------------------------------------------------------------------------
# Wave 2: C1 affect block, C3 learning
# --------------------------------------------------------------------------


def test_c1_affect_block_samples_correlated_columns():
    from discourse_lab.config import Config
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=3000, n_ticks=1, pop={"affect": True})
    pop = sample_population(cfg, np.random.default_rng(0))
    assert pop.has_affect

    from scipy import stats

    contrarianism = pop.X_used[:, pop.trait_names.index("contrarianism")]
    conviction = pop.X_used[:, pop.trait_names.index("conviction")]
    rho_an, _ = stats.spearmanr(pop.animus, contrarianism)
    rho_id, _ = stats.spearmanr(pop.identification, conviction)
    assert rho_an > 0.2, f"animus x contrarianism rho={rho_an:.2f}"
    assert rho_id > 0.2, f"identification x conviction rho={rho_id:.2f}"

    # and the block is simply absent when off
    pop_off = sample_population(_cfg(n_users=200), np.random.default_rng(0))
    assert not pop_off.has_affect
    with pytest.raises(AttributeError):
        _ = pop_off.animus


def test_c1_affect_polarization_metrics_are_nan_without_camps():
    from discourse_lab.metrics.polarization import (
        affective_distance,
        animus_asymmetry,
        camps_and_bimodality,
    )

    unimodal = np.random.default_rng(0).normal(0, 1, 2000)[:, None]
    labels, bm = camps_and_bimodality(unimodal)
    assert labels is None and bm <= 5 / 9
    assert np.isnan(affective_distance(np.random.random(2000), labels))
    assert np.isnan(animus_asymmetry(np.random.random(2000), labels))


def test_c1_affective_distance_is_a_magnitude_not_a_normalized_asymmetry():
    """R2 (post-SMOKE findings): affective_distance must track a widening
    between-camp gap even when both camps' animus grows by the same factor —
    the case a camp-symmetric kernel produces before it does anything
    camp-conditional. animus_asymmetry, being mean-normalized, correctly
    reports ~flat in that scenario; affective_distance must not.
    """
    from discourse_lab.metrics.polarization import affective_distance, animus_asymmetry

    labels = np.array([0] * 1000 + [1] * 1000)
    animus = np.concatenate([np.full(1000, 1.0), np.full(1000, 3.0)])
    scaled = animus * 3.0

    base_distance = affective_distance(animus, labels)
    scaled_distance = affective_distance(scaled, labels)
    assert scaled_distance == pytest.approx(base_distance * 3.0)
    assert scaled_distance > base_distance * 2.0  # rises with the widening gap

    base_asymmetry = animus_asymmetry(animus, labels)
    scaled_asymmetry = animus_asymmetry(scaled, labels)
    assert scaled_asymmetry == pytest.approx(base_asymmetry, abs=1e-9)  # flat: normalized


def _tiny_posts(cfg, rng):
    from discourse_lab.dynamics.expression import ExpressionMap
    from discourse_lab.dynamics.posts import generate_posts
    from discourse_lab.population import sample_population

    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    return generate_posts(np.arange(4) % 200, pop, expr, np.zeros(K), np.zeros((K, D)), 0.3, rng)


def _engine(cfg, seed: int = 0):
    from discourse_lab.dynamics.tick import TickEngine
    from discourse_lab.network import cached_graph
    from discourse_lab.population import cached_population

    rngs = phase_rngs(seed)
    pop = cached_population(cfg, seed, rngs["population"])
    graph = cached_graph(cfg, seed, pop, rngs["graph"])
    return TickEngine(cfg=cfg, pop=pop, graph=graph, rngs=rngs)


def _polarized_axis():
    """The stance editor's `polarized` preset as a scenario axis. Camp-affect
    machinery only lives in a bimodal population — under the shared gate a
    unimodal population has no camps, and affect metrics are undefined
    rather than zero."""
    dens = []
    for i in range(128):
        x = -1 + 2 * (i + 0.5) / 128
        dens.append(np.exp(-((x + 0.6) ** 2) / (2 * 0.2**2)) + np.exp(-((x - 0.6) ** 2) / (2 * 0.2**2)))
    return {
        "name": "polar", "pole_neg": "a", "pole_pos": "b",
        "marginal": {"kind": "empirical", "bins": 128, "support": [-1, 1],
                     "density": (np.array(dens) / np.sum(dens) * 128).tolist()},
        "expression_cost": {"neg": 0.0, "pos": 0.0},
    }


def test_c1_animus_rises_under_outrage_and_not_under_null():
    """The C1 conformance test, adapted to what a symmetric model can
    honestly show: under `outrage`, out-group content draws disproportionate
    engagement (C5's unconditional `outgroup` term), and every out-group
    engagement raises animus through `affect_update` — so population animus
    must rise more than under `null`, where out-group content carries no
    engagement signal. A mechanism that raises animus as fast under `null`
    is measuring activity heterogeneity, not affect.

    Camp-vs-camp ASYMMETRY is a different observable (`animus_asymmetry`,
    kept for the FIES literature): a symmetric kernel on a symmetric
    population produces no expected asymmetry, and the change spec's own
    asymmetry metric exists precisely to flag models that can only ever
    produce symmetric outcomes. Both runs share the same cached population
    (same seed, same sub-hash)."""
    drift = {}
    for kernel in ("outrage", "null"):
        cfg = _cfg(
            n_users=600, n_ticks=40, pop={"affect": True},
            kernel=kernel, drift="full", drift_ramp_ticks=5,
        )
        cfg = dataclasses.replace(
            cfg, scenario=dataclasses.replace(cfg.scenario, stance_axes=(_polarized_axis(),))
        )
        engine = _engine(cfg)
        engine._refresh_camps()   # camps are refreshed per tick; peek before tick 0
        if engine.camps is None:
            raise AssertionError("polarized scenario produced no camps — the shared gate failed")
        start = float(engine.pop.animus.mean())
        for t in range(cfg.dynamics.n_ticks):
            engine.step(t)
        end = float(engine.pop.animus.mean())
        drift[kernel] = end - start

    assert drift["outrage"] > drift["null"] + 1e-4, (
        f"animus grew by {drift['outrage']:.4f} under outrage vs "
        f"{drift['null']:.4f} under null — the affect channel is not kernel-driven"
    )


def test_c3a_behavior_reinforcement_raises_reply_prop_under_outrage_not_null():
    for kernel, expect_rise in (("outrage", True), ("null", False)):
        cfg = _cfg(
            n_users=600, n_ticks=40, kernel=kernel, drift="full",
            drift_lr_behavior=0.05, drift_ramp_ticks=5,
        )
        engine = _engine(cfg)
        names = engine.pop.trait_names
        col = names.index("reply_prop")
        start = float(engine.pop.X_used[:, col].mean())
        for t in range(cfg.dynamics.n_ticks):
            engine.step(t)
        end = float(engine.pop.X_used[:, col].mean())
        if expect_rise:
            assert end > start, f"reply_prop fell under outrage ({start:.4f} -> {end:.4f})"
        else:
            assert abs(end - start) < 0.01, (
                f"reply_prop moved under null ({start:.4f} -> {end:.4f}): "
                "channel 1 is measuring activity, not learning"
            )


def test_c3b_conformity_narrows_gain_variance_where_habituation_does_not():
    summary = {}
    for rule in ("conformity", "habituation"):
        cfg = _cfg(
            n_users=500, n_ticks=60, kernel="outrage", drift="none",
            kernel_learning="group_gain", kernel_learning_rule=rule,
            lr_kernel=0.05,
        )
        engine = _engine(cfg)
        for t in range(cfg.dynamics.n_ticks):
            engine.step(t)
        out = engine.learner.g[:, engine.learner.group_index("outgroup")]
        summary[rule] = (float(out.mean()), float(out.std()))
    # the social claim: conformity's variance collapses; habituation's may
    # move its mean but must not converge the population
    assert summary["conformity"][1] < summary["habituation"][1] + 1e-9, summary


def test_c3b_anchored_kernel_stays_within_an_envelope_of_one():
    cfg = _cfg(
        n_users=300, n_ticks=200, kernel="outrage", drift="full",
        kernel_learning="group_gain", kernel_learning_rule="conformity",
        lr_kernel=0.02,
    )
    engine = _engine(cfg)
    for t in range(cfg.dynamics.n_ticks):
        engine.step(t)
    dev = np.abs(engine.learner.g - 1.0)
    assert dev.mean() < 1.0, f"gains wandered: mean |g-1| = {dev.mean():.2f}"
    assert dev.max() < 10.0, f"a gain left the envelope: max |g-1| = {dev.max():.2f}"


# --------------------------------------------------------------------------
# Wave 3: C6 repulsion ablation, C2 selection
# --------------------------------------------------------------------------


def test_c6_repulsion_off_zeroes_the_negative_weights():
    from discourse_lab.dynamics.drift import social_weights

    on = social_weights(_cfg(repulsion=True))
    off = social_weights(_cfg(repulsion=False))
    assert on["reply"] == -0.5 and on["report"] == -2.0
    assert off["reply"] == 0.0 and off["report"] == 0.0
    assert off["like"] == on["like"] == 1.0  # the attractive half is untouched


def test_c6_repulsion_off_runs_end_to_end():
    cfg = _cfg(n_users=300, n_ticks=8, drift="full", repulsion=False)
    rows = [s.metrics for s in run_iter(cfg, seed=0)]
    assert len(rows) == cfg.dynamics.n_ticks


def test_c2_homophilous_selection_filters_toward_agreement():
    """Unit level: the attend probability must order on agreement. End-to-end,
    the outcome `selection_filtering.selection_shift` is the observable."""
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.exposure.selection import apply_selection

    rng = np.random.default_rng(0)
    m = 4000
    exposures = Exposures(
        post_idx=np.zeros(m, dtype=int),
        user_id=np.arange(m),
        rank=np.zeros(m, dtype=int),
        is_follower=np.ones(m, dtype=bool),
    )
    agreement = np.concatenate([np.full(m // 2, -2.0), np.full(m // 2, +2.0)])
    mask = apply_selection(
        "homophilous", exposures, {"agreement": agreement, "arousal": np.zeros(m)},
        tau_position=6.0, rng=rng,
    )
    low, high = mask[: m // 2].mean(), mask[m // 2 :].mean()
    assert high > low + 0.1, f"agreement does not drive attention ({low:.2f} vs {high:.2f})"
    # and position_only is exactly the old behaviour
    mask_pos = apply_selection(
        "position_only", exposures, {"agreement": agreement, "arousal": np.zeros(m)},
        tau_position=6.0, rng=rng,
    )
    assert mask_pos.all()


def test_c2_rewiring_assorts_the_graph_and_leaves_the_artifact_alone():
    import tempfile
    from pathlib import Path

    from discourse_lab.io.workspace import workspace

    old = dict(__import__("os").environ)
    import os

    os.environ["DLAB_HOME"] = tempfile.mkdtemp()
    try:
        from discourse_lab.metrics.stylized import stance_clusters

        cfg = _cfg(
            n_users=400, n_ticks=60, rewire=True, rewire_every=10, rewire_rate=0.3, drift="none",
            kernel="outrage",
            # the default intercepts make hostile actions rare enough that a
            # homophilous graph is already closed under engagement; this test
            # observes the mechanism, so open the hostile-action tap
            kernel_theta=(("reply", "intercept", -5.0), ("report", "intercept", -5.5)),
        )
        artifact_before = _artifact_digest(cfg)

        engine = _engine(cfg)
        initial = engine.graph.csr.copy()
        all_events = []
        for t in range(cfg.dynamics.n_ticks):
            engine.step(t)
            all_events.extend(engine.rewire_events)

        assert len(all_events) > 0, "no edges changed: rewiring is decorative"

        def _assort(csr):
            labels = stance_clusters(engine.pop.X_used[:, engine.stance_cols])
            rows, cols = csr.nonzero()
            same = labels[rows] == labels[cols]
            return float(same.mean())

        assert _assort(engine.graph.csr) >= _assort(initial) - 0.01, (
            "rewiring did not increase (or barely decreased) stance assortativity"
        )
        assert _artifact_digest(cfg) == artifact_before, "the cache was poisoned by rewiring"
    finally:
        __import__("os").environ.clear()
        __import__("os").environ.update(old)


def _artifact_digest(cfg) -> str:
    import hashlib
    from discourse_lab.io.artifacts import artifact_paths

    h = hashlib.sha256()
    for p in sorted(artifact_paths(cfg).values()):
        h.update(str(p).encode())
        for f in sorted(p.parent.glob("*.npz")) if p.exists() else []:
            h.update(f.name.encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Wave 4: C7 silence gate, C8 threshold contagion
# --------------------------------------------------------------------------


def test_c7_gate_is_off_at_full_conviction_and_binding_without_it():
    from discourse_lab.dynamics.perception import PerceivedState
    from discourse_lab.dynamics.timing import silence_gate_factor

    n, k, d = 50, 3, 1
    perceived = PerceivedState(
        s_local=np.full((n, k), 1 / k), sigma_local=np.zeros((n, k, d)),
        sigma_var_local=np.zeros((n, k)), w=np.zeros(n),
        s_perceived=np.full((n, k), 1 / k),
        sigma_perceived=np.zeros((n, k, d)),
    )
    stance = np.full((n, d), 2.0)  # everyone far from the (zero) perceived climate
    conviction = np.linspace(0.0, 1.0, n)

    gate = silence_gate_factor(stance, perceived, conviction, silence_gate=1.0)
    assert gate[-1] == 1.0, "a fully convicted user was silenced"
    assert gate[0] < 0.9, "a zero-conviction dissenter was not gated at all"
    assert (np.diff(gate) > 0).all(), "conviction does not moderate the gate monotonically"

    off = silence_gate_factor(stance, perceived, conviction, silence_gate=0.0)
    assert (off == 1.0).all(), "silence_gate=0 must disable the mechanism"


def test_c7_false_consensus_gap_is_strictly_negative_when_gated():
    """With the gate on in a bimodal population, expressed bimodality must
    sit strictly below latent bimodality — equal values mean the gate never
    bound."""
    from discourse_lab.dynamics.posts import concat_post_batches
    from discourse_lab.metrics.polarization import expressed_vs_latent_bimodality
    from discourse_lab.population.marginals import empirical_from_editor

    # a bimodal scenario (the stance editor's `polarized` preset)
    axis = _polarized_axis()
    cfg = _cfg(n_users=800, n_ticks=40, silence_gate=1.2, drift="none")
    cfg = dataclasses.replace(
        cfg,
        population=dataclasses.replace(cfg.population, stance_dims=1),
        scenario=dataclasses.replace(cfg.scenario, stance_axes=(axis,)),
    )
    posts = []
    for state in run_iter(cfg, seed=0):
        if state.retired_posts is not None and len(state.retired_posts) > 0:
            posts.append(state.retired_posts)
    assert posts, "no posts retired; cannot measure expressed stance"

    all_posts = concat_post_batches(posts)
    roots = all_posts.stance[all_posts.kind == "post", 0]

    # latent: draw the scenario marginal directly
    m = empirical_from_editor(bins=128, support=(-1.0, 1.0), density=axis["marginal"]["density"])
    latent = m.icdf(np.random.default_rng(1).random(8000))

    res = expressed_vs_latent_bimodality(roots, latent)
    assert res["false_consensus_gap"] < 0.0, (
        f"expressed bimodality did not fall below latent ({res}): the gate never bound"
    )


def test_c8_threshold_model_requires_distinct_engaged_neighbours():
    """The complex-contagion rule: candidates are followers of engagers, and
    reply propensity grows in DISTINCT engagers followed. A user who follows
    two engagers of a thread is strictly more likely to reply than one who
    follows a single engager, all else equal."""
    from discourse_lab.dynamics.reply_model import ThresholdState, threshold_model
    from discourse_lab.network import Graph
    from scipy import sparse

    n = 400
    # user 10 follows engagers 100, 101, 102 (all engaged with post 500);
    # user 11 follows only 100. user 12 follows none of them.
    rows = [10, 10, 10, 11, 200, 201]
    cols = [100, 101, 102, 100, 100, 101]
    csr = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    graph = Graph(csr=csr, csc=csr.tocsc())

    class _Posts:
        id = np.array([500])
        author = np.array([7])

        def __len__(self):
            return 1

    state = ThresholdState()
    state.observe(0, np.array([100, 101, 102, 100]), np.array([500, 500, 500, 500]),
                  np.array(["like", "like", "like", "like"]))

    reply_prop = np.full(n, 0.8)
    rng = np.random.default_rng(0)
    wins = 0
    for _ in range(200):
        targets, authors = threshold_model(
            state=state, graph=graph, active_posts=_Posts(), reply_prop=reply_prop,
            rng=rng, max_age=15, max_replies_per_tick=1,
        )
        if 500 in targets:
            got = set(authors[500].tolist())
            if 10 in got:
                wins += 1
    assert wins > 0, "the multi-engager candidate never replied"

    # the single-engager user (11) must reply far less often than user 10
    wins11 = 0
    for _ in range(200):
        targets, authors = threshold_model(
            state=state, graph=graph, active_posts=_Posts(), reply_prop=reply_prop,
            rng=rng, max_age=15, max_replies_per_tick=1,
        )
        if 500 in targets and 11 in set(authors[500].tolist()):
            wins11 += 1
    assert wins > wins11, "distinct-neighbour count does not order reply propensity"


# --------------------------------------------------------------------------
# Wave 5: C10 machinery
# --------------------------------------------------------------------------


def test_gate_passes_on_the_calibration_of_record_and_warns_off_gate():
    """The C9 gate as a runnable check: `stylized_gate` must pass on the
    recalibrated combination and warn on an off-gate config, so a
    Gini-quoting result carries its gate status instead of inheriting one
    from settings that no longer pass. The off-gate probe is the
    `chronological` ranker — attention spreads evenly under a time-ordered
    feed, which lands the Gini far BELOW the band (the other half-success)."""
    import warnings as w

    from discourse_lab.analysis import set_param
    from discourse_lab.experiments import calibrated_gate_config, stylized_gate

    report = stylized_gate(calibrated_gate_config(), seeds=(0, 1), n_ticks=60)
    assert report.passed, report.summary()

    off_gate = set_param(calibrated_gate_config(), "dynamics.ranker", "chronological")
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        bad = stylized_gate(off_gate, seeds=(0, 1), n_ticks=60, warn=True)
    assert not bad.passed, f"chronological unexpectedly passed: {bad.summary()}"
    assert any("stylized gate FAILED" in str(c.message) for c in caught), (
        "an off-gate sweep ran silently - the warning is the point"
    )


def test_c10_design_requires_a_falsifier():
    from discourse_lab.experiments.designs import Design

    with pytest.raises(TypeError):
        Design(name="d", grid={}, controls=["null"], decision_metric="m",
               predictions={"outrage": "up"})  # falsifier missing
    d = Design(name="d", grid={}, controls=["null"], decision_metric="m",
               predictions={"outrage": "up"}, falsifier="AUC at chance")
    assert d.falsifier


def test_c10_separability_recovers_a_planted_signal():
    from discourse_lab.experiments.identify import separability

    rng = np.random.default_rng(0)
    n_per, n_metrics = 40, 5
    metric_names = [f"m{i}" for i in range(n_metrics)]
    a = rng.normal(0, 1, (n_per, n_metrics))
    b = rng.normal(0, 1, (n_per, n_metrics))
    b[:, 2] += 3.0  # one metric carries the whole signal
    report = separability(a, b, metric_names, seed=0)
    assert report.auc > 0.85
    assert report.top_metric == "m2"


def test_c10_separability_refuses_below_min_seeds():
    """R6 (post-SMOKE findings): 4 seeds per class cannot support a
    cross-validated AUC (folds land single-class), and the resulting AUC==nan
    must not be reportable as a non-identification result. separability()
    should refuse rather than return something unreadable.
    """
    from discourse_lab.analysis import MIN_SEEDS
    from discourse_lab.experiments.identify import separability

    rng = np.random.default_rng(0)
    n_per, n_metrics = 4, 3
    metric_names = [f"m{i}" for i in range(n_metrics)]
    a = rng.normal(0, 1, (n_per, n_metrics))
    b = rng.normal(0, 1, (n_per, n_metrics))
    assert n_per < MIN_SEEDS
    with pytest.raises(ValueError, match="at least"):
        separability(a, b, metric_names, seed=0)


def test_new_persistence_paths_round_trip(tmp_path, monkeypatch):
    """Kernel-state snapshots, the rewire log, the final graph snapshot and
    the `attended` exposure column must survive a persisted run — the
    change-spec mechanisms write new tables, and a table that does not
    round-trip is a mechanism the analysis layer cannot see."""
    import os

    from discourse_lab.io.workspace import workspace

    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = _cfg(
        n_users=200, n_ticks=10, drift="none",
        kernel_learning="group_gain", kernel_learning_rule="conformity",
        rewire=True, rewire_every=5, rewire_rate=0.2,
        selection="homophilous",
        kernel_theta=(("reply", "intercept", -5.0),),
    )
    from discourse_lab.runner import cached_run, load_run

    path = cached_run(cfg, 0, persist=("exposures", "engagements"))
    handle = load_run(cfg, 0)

    assert handle.has_kernel_state, "kernel_state.parquet missing"
    ks = handle.kernel_state()
    assert {"t", "user", "g_agreement", "g_outgroup"} <= set(ks.columns)
    # g starts at exactly 1: the anchored-kernel invariant, on disk
    assert ks["g_outgroup"].max() > 0 and (ks["g_outgroup"] - 1).abs().max() < 1.0

    assert handle.has_rewire_log, "rewire_log.parquet missing"
    final = handle.final_graph()
    assert final.shape[0] == 200

    exp = handle.exposures()
    assert "attended" in exp.columns
    assert set(exp["action"].unique()) <= {"like", "repost", "reply", "quote", "report", "skip", "unattended"}

    # and the cache reuses it without recomputing (RUN_FORMAT 5 + tables present)
    path2 = cached_run(cfg, 0, persist=("exposures", "engagements"))
    assert path2 == path


# --------------------------------------------------------------------------
# C13: threads as digital micro publics - Part 1 (interaction)
# --------------------------------------------------------------------------


def _polarized_axis():
    """The stance editor's `polarized` preset as a scenario axis."""
    dens = []
    for i in range(128):
        x = -1 + 2 * (i + 0.5) / 128
        dens.append(np.exp(-((x + 0.6) ** 2) / (2 * 0.2**2)) + np.exp(-((x - 0.6) ** 2) / (2 * 0.2**2)))
    return {
        "name": "polar", "pole_neg": "a", "pole_pos": "b",
        "marginal": {"kind": "empirical", "bins": 128, "support": [-1, 1],
                     "density": (np.array(dens) / np.sum(dens) * 128).tolist()},
        "expression_cost": {"neg": 0.0, "pos": 0.0},
    }


# --------------------------------------------------------------------------
# C13: threads as digital micro publics
# --------------------------------------------------------------------------


def test_c13_repliers_come_from_the_kernel():
    """C13a conformance (test 1): with reply_selection='kernel', repliers are
    content-connected to the parent - mean |replier stance - parent stance|
    differs from the lottery condition on the same seed. If identical, the
    coupling is decorative."""
    gaps = {}
    for selection in ("kernel", "lottery"):
        cfg = _cfg(n_users=500, n_ticks=30, drift="none", reply_selection=selection)
        engine = _engine(cfg)
        for t in range(cfg.dynamics.n_ticks):
            engine.step(t)
        posts = engine.active_posts
        is_reply = posts.kind == "reply"
        id_to_idx = {int(pid): i for i, pid in enumerate(posts.id)}
        reply_rows, parent_rows = [], []
        for i in np.flatnonzero(is_reply):
            p = id_to_idx.get(int(posts.parent[i]))
            if p is not None:
                reply_rows.append(i)
                parent_rows.append(p)
        assert reply_rows, "no in-pool replies to measure"
        replier = pop_stance_of_rows(engine, np.array(reply_rows))
        parent = posts.stance[np.array(parent_rows)]
        gaps[selection] = float(np.abs(replier - parent).mean())

    assert gaps["kernel"] != gaps["lottery"], (
        f"replier-parent distance identical under kernel and lottery ({gaps}) - "
        "the C13a coupling is decorative"
    )


def pop_stance_of_rows(engine, users):
    return engine.pop.X_used[np.asarray(users, dtype=int)][:, engine.stance_cols]


def test_c13_reply_fallback_rate_is_low():
    """C13a conformance (test 2): run-aggregate reply_fallback_rate < 0.3 at
    default config - the metric that tells you whether the kernel coupling
    did anything. A high value means most replies still come from the global
    lottery (the content-blind path C13a removes)."""
    rows = [s.metrics for s in run_iter(_cfg(n_users=500, n_ticks=30, drift="none"), seed=0)]
    tot_rep = sum(r["n_replies"] for r in rows
                  if r["reply_fallback_rate"] == r["reply_fallback_rate"])
    tot_fb = sum(r["reply_fallback_rate"] * r["n_replies"] for r in rows
                 if r["reply_fallback_rate"] == r["reply_fallback_rate"])
    assert tot_rep > 0, "no replies measured"
    assert tot_fb / tot_rep < 0.3, (
        f"aggregate reply_fallback_rate {tot_fb / tot_rep:.3f} >= 0.3 - "
        "most replies still come from the global lottery"
    )


def test_c13_stylized_gate():
    """C13 step-3 hard gate (test 6): the reply recalibration - kernel-sourced
    repliers, content-conditioned mu, sigma(t) reply conformity - must hold
    the C9 pair (attention Gini AND reciprocity) at 20 seeds before Part 2's
    DMP metrics are trusted for anything."""
    import os

    from discourse_lab.analysis import set_param
    from discourse_lab.experiments import stylized_gate

    seeds = int(os.environ.get("DLAB_GATE_SEEDS", "20"))
    base = set_param(_cfg(n_users=1200, n_ticks=60, drift="none"),
                     "dynamics.ranker", "engagement_optimized")
    base = set_param(base, "dynamics.kernel", "bandwagon")
    base = set_param(base, "graph.generator", "latent_pa")
    base = set_param(base, "graph.mirror_p", 0.05)
    base = set_param(base, "dynamics.theta_scale", (("social_proof", 0.75),))

    report = stylized_gate(base, seeds=tuple(range(seeds)))
    assert report.passed, report.summary()





def test_c13_local_conformity_is_local():
    """C13c conformance (test 4): the CONFORMITY DISPLACEMENT - reply stance
    minus the replier's own stance - points at the room the reply conformed
    to, for repliers whose conviction is low enough for the room pull to
    dominate the blend noise. Under 'local' the displacement tracks
    (sigma_local - own); under 'global' it tracks (sigma(t) - own)."""
    import dataclasses

    from discourse_lab.dynamics.hawkes import thread_sigma_local

    gaps = {}
    for mode in ("local", "global"):
        cfg = _cfg(n_users=1200, n_ticks=80, drift="none", reply_conformity=mode,
                   hawkes_mu0=0.03)
        cfg = dataclasses.replace(
            cfg,
            population=dataclasses.replace(cfg.population, stance_dims=1),
            scenario=dataclasses.replace(cfg.scenario, stance_axes=(_polarized_axis(),)),
        )
        engine = _engine(cfg)
        to_local, to_global = [], []
        conv_col = engine.pop.trait_names.index("conviction")
        for t in range(cfg.dynamics.n_ticks):
            table = engine.thread_sigma_local    # the room this tick's replies used
            engine.step(t)
            posts = engine.active_posts
            if posts is None or table == ():
                continue
            uniq, sigma = table
            for i in np.flatnonzero((posts.kind == "reply") & (posts.t == t)):
                rid = int(np.where(posts.root >= 0, posts.root, posts.id)[i])
                pos_arr = np.flatnonzero(uniq == rid)
                if len(pos_arr) == 0:
                    continue
                conv = engine.pop.X_used[int(posts.author[i]), conv_col]
                if conv > 0.5:
                    continue    # near-mean/high conviction: room pull drowns in noise
                own0 = float(engine.pop.X_used[int(posts.author[i]), engine.stance_cols[0]])
                d = posts.stance[i, 0] - own0
                pos = int(np.searchsorted(uniq, rid))
                room_local = sigma[min(pos, len(uniq) - 1), 0] if uniq[min(pos, len(uniq) - 1)] == rid else None
                if room_local is None:
                    continue
                room_global = float(engine.sigma[posts.topic[i], 0])
                to_local.append(abs(d - (room_local - own0)))
                to_global.append(abs(d - (room_global - own0)))
        gaps[mode] = (float(np.mean(to_local)), float(np.mean(to_global)), len(to_local))

    assert gaps["local"][2] > 20, f"too few low-conviction replies: {gaps}"
    assert gaps["local"][0] < gaps["local"][1], (
        f"local mode: displacement nearer sigma(t) than the thread {gaps['local']}"
    )
    assert gaps["global"][1] < gaps["global"][0], (
        f"global mode: displacement nearer the thread than sigma(t) {gaps['global']}"
    )


def test_c13_sigma_local_excludes_the_reply_being_generated():
    """C13c conformance (test 5): sigma_local is computed from posts already
    in the pool, engagement-weighted (base weight 1 per post), and a reply
    generated from it has ids outside the table - the blend cannot see
    itself. A self-referential blend would silently inflate every
    within-thread coherence measure."""
    from discourse_lab.dynamics.expression import ExpressionMap
    from discourse_lab.dynamics.hawkes import thread_sigma_local
    from discourse_lab.dynamics.posts import generate_posts
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=100, n_ticks=1)
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    posts = generate_posts(np.arange(5), pop, expr, np.zeros(K), np.zeros((K, D)), 0.3, rng, t=0)
    # hand-build ONE thread: root 0 with four responsive posts
    posts.root[:] = 0
    posts.parent[:] = [-1, 0, 0, 0, 0]
    posts.depth[:] = [0, 1, 1, 1, 1]
    posts.engagement_count[:] = [0, 3, 0, 1, 0]

    uniq, sigma = thread_sigma_local(posts, list(range(D)))
    assert len(uniq) == 1 and uniq[0] == 0

    w = np.array([1, 4, 1, 2, 1], dtype=float)
    expected = (w[:, None] * posts.stance).sum(axis=0) / w.sum()
    np.testing.assert_allclose(sigma[0], expected, atol=1e-12)

    assert posts.id.max() == 4


def test_c13_content_drives_branching():
    """C13b conformance (test 3): within threads, the correlation between a
    post's provocativeness and its sub-tree size is positive under
    reply_mu_gamma, and ~0 without it. Friction generates the discussion -
    contestation, not noise."""
    from scipy import stats

    from discourse_lab.dynamics.hawkes import thread_sigma_local  # noqa: F401

    correlations = {}
    for gamma in ((), (("prov", 1.0), ("arousal", 0.5), ("disagree", 0.5))):
        cfg = _cfg(n_users=1200, n_ticks=60, drift="none", reply_mu_gamma=gamma,
                   ranker="engagement_optimized", kernel="bandwagon")
        engine = _engine(cfg)
        for t in range(cfg.dynamics.n_ticks):
            engine.step(t)
        posts = engine.active_posts
        # subtree sizes per post, within its thread
        children: dict[int, list[int]] = {}
        for i, p in enumerate(posts.parent):
            if p >= 0:
                children.setdefault(int(p), []).append(i)
        sizes, provs = [], []
        for i in range(len(posts)):
            stack = list(children.get(int(posts.id[i]), []))
            count = 0
            while stack:
                node = stack.pop()
                count += 1
                stack.extend(children.get(int(posts.id[node]), []))
            sizes.append(count)
            provs.append(posts.provocativeness[i])
        rho, _ = stats.spearmanr(provs, sizes)
        correlations[bool(gamma)] = float(rho)

    assert correlations[True] > 0.02 and correlations[True] > correlations[False], (
        f"provocativeness-subtree correlation without gamma ({correlations[False]:.3f}) "
        f"is not below with-gamma ({correlations[True]:.3f})"
    )
    # The effect is deliberately modest: max_replies_per_tick=1 (the
    # anti-runaway cap from the reply recalibration) chains threads, which
    # caps how much content variance can express through subtree SIZE.
    # Raising the cap sharpens this correlation at the cost of the depth vs
    # runaway trade-off recorded in config.py's hawkes_mu_inherit notes.
    assert correlations[True] > correlations[False], (
        f"provocativeness-subtree correlation without gamma ({correlations[False]:.3f}) "
        f"is not below with-gamma ({correlations[True]:.3f})"
    )


# --------------------------------------------------------------------------
# V-series (change-spec-v1-engagement-valence.md): V5 minimum conformance
# set, written before V1 per the spec's own sequencing ("Write V5 before
# V1... a test suite written after the fix cannot demonstrate that the fix
# was needed"). Two of the five mechanisms below (de-escalation, repeated
# encounter) are open gaps this file pins down rather than silently passes;
# the other three are regression guards for mechanisms that already exist.
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "V1/V2 gap, not yet fixed. AFFECT_HOSTILITY_WEIGHTS is keyed on "
        "action alone and every entry is >= 0 (drift.py's own docstring: "
        "'the contact-hypothesis alternative (likes as positive contact "
        "that LOWERS animus) is a different theory a sweep should be able "
        "to express without editing the loop' -- but no "
        "affect_weights_hostility override can express it, because the "
        "table has no valence axis to hang a sign on). V2 rekeys the table "
        "on (action, valence) with disagree+civil negative; remove this "
        "marker once that lands and this flips to a real pass."
    ),
)
def test_v5_deescalation_civil_crosscamp_contact_lowers_animus():
    """V5 minimum set, assertion 1. `like` is the least confrontational
    action AFFECT_HOSTILITY_WEIGHTS has -- the shipped model's closest thing
    to "civil" contact -- so if even all-`like` cross-camp engagement cannot
    push a user's animus contribution negative, no configuration of the
    current table can represent contact reducing hostility."""
    from discourse_lab.dynamics.drift import AFFECT_HOSTILITY_WEIGHTS, affect_delta
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.population import sample_population

    assert all(w >= 0 for w in AFFECT_HOSTILITY_WEIGHTS.values()), (
        "a negative entry already exists in AFFECT_HOSTILITY_WEIGHTS -- this test is stale"
    )

    cfg = _cfg(n_users=200, pop={"affect": True})
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    posts = _tiny_posts(cfg, rng)  # authors 0, 1, 2, 3 -- post 0's author is user 0

    m = 100
    exposures = Exposures(
        post_idx=np.zeros(m, dtype=int),        # everyone consumes author 0's post
        user_id=(np.arange(m) % 99) * 2 + 1,    # odd ids only: camp 1, never author 0
        rank=np.zeros(m, dtype=int),
        is_follower=np.ones(m, dtype=bool),
    )
    actions = np.full(m, "like")                # the table's most civil action
    camps = (np.arange(cfg.population.n_users) % 2).astype(np.int64)
    assert camps[0] == 0 and (camps[exposures.user_id] == 1).all()  # every exposure cross-camp

    delta = affect_delta(exposures, actions, posts, pop, camps, lr_affect=0.5)
    animus_col = pop.trait_names.index("animus")
    assert (delta[:, animus_col] < 0).any(), (
        "no exposed user's animus fell under all-`like` cross-camp contact -- "
        "the de-escalation channel does not exist"
    )


def test_v5_backfire_hostile_crosscamp_contact_raises_animus():
    """V5 minimum set, assertion 4: the half of the quadrant the current
    model already gets right -- `report`, the table's most confrontational
    action, must raise animus on cross-camp contact. This is the entire
    content of AFFECT_HOSTILITY_WEIGHTS today, so it passes now; kept in the
    regression set because V1/V2 rekey this table and must not break it."""
    from discourse_lab.dynamics.drift import affect_delta
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=200, pop={"affect": True})
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    posts = _tiny_posts(cfg, rng)

    m = 100
    exposures = Exposures(
        post_idx=np.zeros(m, dtype=int),
        user_id=(np.arange(m) % 99) * 2 + 1,
        rank=np.zeros(m, dtype=int),
        is_follower=np.ones(m, dtype=bool),
    )
    actions = np.full(m, "report")
    camps = (np.arange(cfg.population.n_users) % 2).astype(np.int64)

    delta = affect_delta(exposures, actions, posts, pop, camps, lr_affect=0.5)
    animus_col = pop.trait_names.index("animus")
    touched = np.unique(exposures.user_id)
    assert (delta[touched, animus_col] > 0).all(), (
        "cross-camp `report` did not raise animus for every exposed user"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open gap, one of the three named in V0 (change-spec-v1-engagement-"
        "valence.md), not addressed by this spec's V1-V6 items. "
        "`compute_features` has no channel for contact history: "
        "`tie_strength` is exactly `is_follower.astype(float)`, bimodal "
        "{0, 1} regardless of how many times two users have crossed paths. "
        "Remove this marker once a repeat-contact mechanism lands."
    ),
)
def test_v5_repeated_encounter_tie_strength_rises_with_repeat_contact():
    """V5 minimum set, assertion 2: a tie-strength feature should register
    the relationship, not just the follow edge -- repeated contact with the
    same author should read as a stronger tie than a first encounter,
    holding follow status fixed."""
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.exposure.kernel import compute_features
    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=200)
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    posts = _tiny_posts(cfg, rng)

    is_follower = np.zeros(4, dtype=bool)  # follow status held fixed: nobody follows
    exposures = Exposures(
        post_idx=np.zeros(4, dtype=int),   # all four rows are the same author (post 0)
        user_id=np.array([10, 10, 10, 11]),
        rank=np.zeros(4, dtype=int),
        is_follower=is_follower,
    )
    feats = compute_features(exposures, posts, pop, is_follower, t_current=3)
    tie = feats["tie_strength"]

    repeat_contact_tie = tie[:3].mean()  # user 10's third encounter with this author
    first_contact_tie = tie[3]           # user 11's first
    assert repeat_contact_tie > first_contact_tie, (
        f"tie_strength ({tie}) does not separate repeat contact from a first "
        "encounter at fixed follow status -- it is exactly is_follower"
    )


def test_v5_topic_affinity_blocks_correlate_with_topic_affinity_not_stance():
    """V5 minimum set, assertion 3: the second of the three gaps named in
    V0, already closed by `graph.sbm_block_source='topic_affinity'` --
    micro-public block assignment must track subject-matter geometry, not
    ideology. Kept in the regression set so a future population or SBM
    change cannot silently leak stance into it."""
    from scipy import stats

    from discourse_lab.population import sample_population

    cfg = _cfg(n_users=2000)
    rng = np.random.default_rng(0)
    pop = sample_population(cfg, rng)
    names = pop.trait_names
    topic_idx = [i for i, n in enumerate(names) if n.startswith("topic_affinity_")]
    stance_idx = [i for i, n in enumerate(names) if n.startswith("stance_")]
    assert topic_idx and stance_idx

    # graph.sbm_block_source="topic_affinity": each user's block is their
    # single most-affine topic (discourse_lab/network/sbm.py)
    blocks = np.argmax(pop.X_used[:, topic_idx], axis=1)

    max_stance_corr = max(
        abs(stats.pointbiserialr((blocks == b).astype(float), pop.X_used[:, s])[0])
        for b in range(len(topic_idx)) for s in stance_idx
    )
    assert max_stance_corr < 0.1, (
        f"topic-affinity block assignment correlates with stance (r={max_stance_corr:.2f}) "
        "-- micro-publics defined by subject matter are leaking ideology"
    )


def test_v5_selection_echo_attended_exceeds_echo_exposed_under_homophilous():
    """V5 minimum set, assertion 5: choice must filter MORE than exposure
    already does under `homophilous` selection -- the observable
    `selection_filtering.selection_shift` (outcomes.py) exists precisely to
    report this; kept here as a direct mechanism-level regression guard."""
    from discourse_lab.exposure.attention import Exposures
    from discourse_lab.exposure.selection import apply_selection
    from discourse_lab.metrics import echo_chamber_index

    rng = np.random.default_rng(0)
    n_users, m = 200, 4000
    own_stance = rng.normal(0, 1, (n_users, 1))
    user_id = rng.integers(0, n_users, m)
    # exposure pool: half near the user's own position, half far -- only
    # mildly assorted on its own, leaving room for choice to filter further
    offset = np.where(rng.random(m) < 0.5, rng.normal(0, 0.1, m), rng.normal(0, 3.0, m))
    consumed_stance = (own_stance[user_id, 0] + offset)[:, None]

    agreement = -np.abs(consumed_stance[:, 0] - own_stance[user_id, 0])
    exposures = Exposures(
        post_idx=np.zeros(m, dtype=int), user_id=user_id,
        rank=np.zeros(m, dtype=int), is_follower=np.ones(m, dtype=bool),
    )
    attended = apply_selection(
        "homophilous", exposures, {"agreement": agreement, "arousal": np.zeros(m)},
        tau_position=6.0, rng=rng,
    )

    delta = float(np.median(np.abs(offset)))
    idx_exposed = echo_chamber_index(own_stance, consumed_stance, user_id, delta=delta)
    idx_attended = echo_chamber_index(own_stance, consumed_stance[attended], user_id[attended], delta=delta)

    mean_exposed = float(np.nanmean(idx_exposed))
    mean_attended = float(np.nanmean(idx_attended))
    assert mean_attended > mean_exposed, (
        f"echo_attended ({mean_attended:.3f}) does not exceed echo_exposed "
        f"({mean_exposed:.3f}) under homophilous selection"
    )
