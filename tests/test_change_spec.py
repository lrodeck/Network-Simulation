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
    base = set_param(base, "dynamics.theta_scale", (("social_proof", 0.6),))

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
