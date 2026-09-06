"""Guards against the failure mode that hid four spec mechanisms for the whole
build: code that is written, tested, exported — and never called.

`dynamics/hawkes.py` implemented spec §2.3's reply scheduler correctly from
step 5 onward and the tick loop never invoked it. It had unit tests, it was in
`dynamics.__all__`, and the suite was green. Nothing in the repo could tell the
difference between a wired mechanism and a decorative one, so a duplicate was
later written alongside it under the same name.

These tests assert that each mechanism spec §3.1's tick loop lists is actually
reached when a tick runs, by observing its effect rather than its definition.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from discourse_lab.config import Config
from discourse_lab.runner import run_iter


def _cfg(n_users: int = 400, n_ticks: int = 12, **dyn):
    return dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=n_users),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=n_ticks, **dyn),
    )


def test_step_2_hawkes_reply_scheduler_is_actually_invoked():
    """§3.1 step 2: `replies = hawkes_draw(open_threads, t)`."""
    kinds = set()
    for state in run_iter(_cfg(drift="none"), seed=0):
        if state.retired_posts is not None and len(state.retired_posts) > 0:
            kinds.update(np.unique(state.retired_posts.kind).tolist())
    assert "reply" in kinds, "no reply posts: the Hawkes scheduler is not wired"


def test_step_6_salient_events_are_queued():
    """§3.1 step 6: `flag_salient_events(engagements)  # queued, not executed`."""
    saw_queue = any(
        len(state.salient_events) > 0
        for state in run_iter(_cfg(n_users=800, n_ticks=20, drift="none"), seed=0)
    )
    assert saw_queue, "no salient events queued: channel 3's trigger is not wired"


def test_step_6_drift_runs_on_a_tick_with_no_engagement():
    """§3.1 puts drift at step 6 unconditionally, and §2.9 says the OU
    mean-reversion term "is not optional". Both sat inside the exposure guard,
    so a quiet tick skipped the reversion while its deltas had already landed.

    Driven directly rather than by hoping for a quiet tick — at realistic
    populations there aren't any, which is exactly why this went unnoticed.
    """
    from discourse_lab.dynamics.tick import TickEngine
    from discourse_lab.network import cached_graph
    from discourse_lab.population import cached_population
    from discourse_lab.runner import phase_rngs

    cfg = _cfg(n_users=300, n_ticks=1, drift="full")
    rngs = phase_rngs(0)
    pop = cached_population(cfg, 0, rngs["population"])
    graph = cached_graph(cfg, 0, pop, rngs["graph"])
    engine = TickEngine(cfg=cfg, pop=pop, graph=graph, rngs=rngs)

    engine.step(0)                      # initialises DriftState.Bs
    engine.active_posts = None          # force a genuinely quiet tick
    before = pop.X_stored.copy()
    engine.step(1)

    assert not np.allclose(pop.X_stored, before), "drift did not run on a quiet tick"


def test_step_7_discourse_state_decays_on_a_quiet_tick():
    """§3.1 step 1 decays `s` every tick regardless of what happened."""
    from discourse_lab.dynamics.tick import TickEngine
    from discourse_lab.network import cached_graph
    from discourse_lab.population import cached_population
    from discourse_lab.runner import phase_rngs

    cfg = _cfg(n_users=300, n_ticks=1, drift="none")
    rngs = phase_rngs(0)
    pop = cached_population(cfg, 0, rngs["population"])
    graph = cached_graph(cfg, 0, pop, rngs["graph"])
    engine = TickEngine(cfg=cfg, pop=pop, graph=graph, rngs=rngs)

    engine.step(0)
    engine.s = np.ones_like(engine.s)   # a non-zero agenda
    engine.active_posts = None
    engine.step(1)

    assert np.all(engine.s < 1.0), "discourse state froze on a quiet tick"


def test_every_config_field_is_read_by_something():
    """The four Hawkes fields were declared and never read for the whole
    build, and so were `posts_per_tick_rate` and `exposure_sample_rate`. A
    declared parameter that nothing consumes is a mechanism that silently
    does not exist.

    Fields listed in KNOWN_UNREAD are deliberate: they are consumed by the
    offline LLM pass, not the tick loop.
    """
    import re
    import subprocess
    from pathlib import Path

    # Each entry is a gap, not an exemption. Delete the entry when the gap
    # closes; do not add one to make this test pass.
    KNOWN_UNREAD = {
        "llm_model",              # threaded into the client via WorldConfig, not by name
        "label",                  # provenance only, never read by the model
        # spec §2.9 channel 3's epsilon in `clip(delta_llm, -eps, eps)`.
        # `parse_adjudication` applies it correctly and `detect_salient_events`
        # now queues the triggers, but nothing calls `request_adjudication` —
        # there is no offline adjudication pass, so channel 3 is queue-only.
        "adjudication_max_delta",
    }

    src = Path("discourse_lab/config.py").read_text(encoding="utf-8")
    fields = set(re.findall(r"^\s{4}(\w+):\s*[\w\[\]\.\| ]+\s*=", src, re.M))

    unread = []
    for name in sorted(fields - KNOWN_UNREAD):
        hits = subprocess.run(
            ["grep", "-rn", rf"\b{name}\b", "discourse_lab", "--include=*.py"],
            capture_output=True, text=True,
        ).stdout.splitlines()
        if not [h for h in hits if not h.startswith("discourse_lab/config.py")]:
            unread.append(name)

    assert not unread, f"config fields declared but never read: {unread}"


def test_reply_inheritance_seeds_decaying_excitation_not_the_baseline():
    """`hawkes_mu_inherit` must warm a reply's thread through `excitation`,
    which decays at beta, and never through `mu`, which does not.

    Seeding `mu` gives every reply a permanently raised floor, so the process
    runs away regardless of alpha/beta. Measured replies/tick over ticks
    0-20 / 40-60 / 100-120 with mu-seeding at ratio=0.6: 1.9 / 10.0 / 400.7.
    """
    from discourse_lab.dynamics.hawkes import HawkesThreads

    th = HawkesThreads()
    th.open_threads(np.array([1]), mu=0.004, excitation=10.0)

    assert th.mu[0] == 0.004, "inherited heat leaked into the permanent baseline"
    assert th.excitation[0] == 10.0

    th.step(np.random.default_rng(0), alpha=0.9, beta=1.5, max_age=50)
    assert th.excitation[0] < 10.0, "inherited heat is not decaying"


def test_default_hawkes_settings_are_subcritical():
    """spec §2.3 requires alpha/beta < 1 for stability and warns that threads
    otherwise run away. The inheritance channel adds to that, so the defaults
    are checked end to end rather than trusted from the ratio alone.
    """
    cfg = _cfg(n_users=600, n_ticks=60, drift="none")
    replies = [state.metrics["n_replies"] for state in run_iter(cfg, seed=0)]

    early = float(np.mean(replies[:15]))
    late = float(np.mean(replies[-15:]))
    assert late < 10 * max(early, 1.0), (
        f"reply volume grew from {early:.1f} to {late:.1f} per tick — "
        "the default Hawkes settings are supercritical"
    )


def test_alpha_bisection_hits_the_target_mean_degree():
    """spec §2.2: "alpha is calibrated by bisection to hit a target mean
    degree". The bisection runs over the kNN candidate draw, but long ties are
    added afterwards, so without correcting for `long_tie_fraction` the graph
    lands (1 + f) over target — measured 22.3 against 20, and 44.5 against 40.

    The residual tolerated here is the reciprocity pass, which §2.2 applies
    after generation as a separate step.
    """
    from discourse_lab.network import generate_graph
    from discourse_lab.population import sample_population

    for target in (20.0, 40.0):
        cfg = dataclasses.replace(
            Config(),
            population=dataclasses.replace(Config().population, n_users=4000),
            graph=dataclasses.replace(Config().graph, mean_degree=target),
        )
        rng = np.random.default_rng(0)
        pop = sample_population(cfg, rng)
        achieved = generate_graph(cfg, pop, rng).csr.nnz / 4000

        assert abs(achieved / target - 1) < 0.05, (
            f"target mean degree {target}, achieved {achieved:.2f}"
        )
