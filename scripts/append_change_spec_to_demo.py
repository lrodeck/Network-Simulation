# Appends the change-spec section (C1-C10) to notebooks/demo.ipynb.
# One-off maintenance script; kept so the section's provenance is auditable.
import json

path = "notebooks/demo.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""## 17. Change-spec mechanisms (C1–C10)

Ten literature-informed mechanisms, **all defaulted off** — nothing above
moved. Each block below turns one on and observes its effect, conformance-test
style: a mechanism that changes nothing is not wired. The prose is in
[MODEL.md §16](../MODEL.md); the conformance suite is
`tests/test_change_spec.py`.
"""))

cells.append(md("""### C1 — the affect block

Ideological position is not affective polarization. `identification`
(attachment to your own camp) and `animus` (hostility toward the other) are a
separate **state** that updates from interaction outcomes — every out-group
engagement raises animus (Rathje et al. 2021), with the weight table kept
deliberately separate from social influence's, where replying is
stance-*repulsive*.

Camp is gated: where the population is unimodal, camps are noise and every
affect number reads `nan`, not zero. So this run uses a polarized scenario.
"""))

cells.append(code("""from discourse_lab.dynamics.tick import TickEngine
from discourse_lab.metrics.polarization import (
    affective_distance, animus_asymmetry, camps_and_bimodality, ident_animus_coupling,
)

# the stance editor's `polarized` preset as a one-axis scenario: camps must
# exist (bimodality > 5/9) for anything camp-conditional to be defined
dens = []
for i in range(128):
    x = -1 + 2 * (i + 0.5) / 128
    dens.append(np.exp(-((x + 0.6) ** 2) / (2 * 0.2**2)) + np.exp(-((x - 0.6) ** 2) / (2 * 0.2**2)))
axis = {"name": "polar", "pole_neg": "camp A", "pole_pos": "camp B",
        "marginal": {"kind": "empirical", "bins": 128, "support": [-1, 1],
                     "density": (np.array(dens) / np.sum(dens) * 128).tolist()},
        "expression_cost": {"neg": 0.0, "pos": 0.0}}

cfg_affect = dataclasses.replace(
    cfg,
    population=dataclasses.replace(cfg.population, n_users=800, affect=True, stance_dims=1),
    dynamics=dataclasses.replace(cfg.dynamics, n_ticks=40, kernel="outrage", drift="full",
                                 drift_ramp_ticks=5, lr_affect=0.05),
    scenario=dataclasses.replace(cfg.scenario, stance_axes=(axis,)),
)

rngs_affect = phase_rngs(SEED)
pop_affect = cached_population(cfg_affect, SEED, rngs_affect["population"])
graph_affect = cached_graph(cfg_affect, SEED, pop_affect, rngs_affect["graph"])
engine_a = TickEngine(cfg=cfg_affect, pop=pop_affect, graph=graph_affect, rngs=rngs_affect)
engine_a._refresh_camps()

labels, bm = camps_and_bimodality(pop_affect.X_used[:, engine_a.stance_cols])
print(f"bimodality {bm:.2f} (gate {5/9:.3f}) -> camps "
      f"{'defined' if labels is not None else 'undefined; affect comparisons are nan'}")

animus_0 = float(pop_affect.animus.mean())
for t in range(cfg_affect.dynamics.n_ticks):
    engine_a.step(t)
animus_1 = float(engine_a.pop.animus.mean())
print(f"mean animus: {animus_0:.4f} -> {animus_1:.4f} under outrage")
print(f"animus asymmetry: {animus_asymmetry(engine_a.pop.animus, engine_a.camps):+.4f} "
      f"(symmetric kernels produce none in expectation - that is what the metric is for)")
print(f"ident-animus coupling (the sorting signature): "
      f"{ident_animus_coupling(pop_affect.identification, engine_a.pop.animus):+.3f}")
"""))

cells.append(md("""### C3 — learnable kernels

`kernel_learning="group_gain"` gives every user a gain over named coefficient
groups: `theta_u = theta_base * g_u`, so `g == 1` **is** the named kernel and
the run stays describable as "outrage, plus this much learned deviation".

The rule is named in the config, never implied: `conformity` (drift toward
followees — Brady et al. 2021's norm convergence) vs `habituation`
(use-driven, no social channel — the control). The social claim is the
*variance collapse* under conformity that habituation does not show.
"""))

cells.append(code("""gain_stats = {}
for rule in ("conformity", "habituation"):
    cfg_learn = dataclasses.replace(
        cfg_affect,
        dynamics=dataclasses.replace(cfg_affect.dynamics, kernel="outrage", drift="none",
                                     kernel_learning="group_gain", kernel_learning_rule=rule,
                                     lr_kernel=0.05),
    )
    pop_learn = cached_population(cfg_learn, SEED, rngs_affect["population"])
    graph_learn = cached_graph(cfg_learn, SEED, pop_learn, rngs_affect["graph"])
    e = TickEngine(cfg=cfg_learn, pop=pop_learn, graph=graph_learn, rngs=phase_rngs(SEED))
    for t in range(cfg_learn.dynamics.n_ticks):
        e.step(t)
    out = e.learner.g[:, e.learner.group_index("outgroup")]
    gain_stats[rule] = (float(out.mean()), float(out.std()))

for rule, (mean, sd) in gain_stats.items():
    print(f"{rule:12s} outgroup gain: mean {mean:+.4f}  sd {sd:.4f}")
print("conformity's sd should sit at or below habituation's - "
      "if it doesn't, the norm-convergence claim is unsupported")
"""))

cells.append(md("""### C2 — selection: choice vs the algorithm

Bakshy, Messing & Adamic (2015) found individual choice filtered cross-cutting
content *more than the algorithm did*. Selection sits between ranking and the
kernel: `P(attend | exposed) = sigma(beta_pos*decay + beta_agree*agreement +
beta_arousal*arousal)`. `position_only` reproduces the old behaviour exactly.

The exposure log now records both states, so the Bakshy decomposition is a
measured quantity, not a decomposition you do in your head:
"""))

cells.append(code("""from discourse_lab.outcomes import selection_filtering

cfg_sel = dataclasses.replace(
    cfg,
    dynamics=dataclasses.replace(cfg.dynamics, selection="homophilous", n_ticks=30),
)
cached_run(cfg_sel, seed=SEED, persist=("posts", "engagements", "exposures"))
handle_sel = load_run(cfg_sel, seed=SEED)

sf = selection_filtering(handle_sel, pop, lex)
print(f"echo-chamber index on everything exposed: {sf['echo_exposed']:.4f}")
print(f"echo-chamber index on attended only:      {sf['echo_attended']:.4f}")
print(f"selection_shift (the choice component):   {sf['selection_shift']:+.4f}")
print("zero shift means choice filtered nothing - the selection stage is decorative")
"""))

cells.append(md("""### C2 — rewiring

Follow/unfollow as a slow process on accumulated interaction valence:
unfollow who was hostile to you, follow whose content you engaged with. It
runs every `rewire_every` ticks (a per-tick sparse rebuild is the one thing
here that could dominate runtime), writes an edge-change log, and never
touches the cached initial graph.
"""))

cells.append(code("""from discourse_lab.metrics.stylized import stance_clusters

cfg_rw = dataclasses.replace(
    cfg,
    population=dataclasses.replace(cfg.population, n_users=400),
    dynamics=dataclasses.replace(cfg.dynamics, rewire=True, rewire_every=10,
                                 rewire_rate=0.3, n_ticks=40, kernel="outrage",
                                 kernel_theta=(("reply", "intercept", -5.0),)),
)
pop_rw = cached_population(cfg_rw, SEED, rngs_affect["population"])
e_rw = TickEngine(cfg=cfg_rw, pop=pop_rw,
                  graph=cached_graph(cfg_rw, SEED, pop_rw, rngs_affect["graph"]),
                  rngs=phase_rngs(SEED))
initial = e_rw.graph.csr.copy()
events = []
for t in range(cfg_rw.dynamics.n_ticks):
    e_rw.step(t)
    events.extend(e_rw.rewire_events)


def assort(csr):
    lab = stance_clusters(e_rw.pop.X_used[:, e_rw.stance_cols])
    r, c = csr.nonzero()
    return float((lab[r] == lab[c]).mean())


print(f"edge changes: {len(events)} over {cfg_rw.dynamics.n_ticks} ticks")
print(f"stance assortativity: {assort(initial):.4f} -> {assort(e_rw.graph.csr):.4f}")
"""))

cells.append(md("""### C4 + C5 — the quality backdoor, closed; the out-group feature, made general

`quality` used to be generated from author traits — the same traits that
drive prominence — so Spearman(quality, engagement) was nonzero even under
`null`. `quality_trait_coupling=0` (the default now) draws quality
independently; `quality_attention_lift` refuses to compute without a matched
null run, because a metric that *can* be computed from one run *will* be.

C5 is the other half of making kernels separable: `outrage` now carries
out-group attraction on an unconditional `outgroup` feature (Rathje et al.
2021: out-group terms raised sharing odds 67%), with `disagree_x_con` demoted
to a modifier. **Pre-change `outrage` results are not comparable.**
"""))

cells.append(code("""from discourse_lab.outcomes import quality_attention_lift

cfg_epi = dataclasses.replace(cfg, dynamics=dataclasses.replace(cfg.dynamics, kernel="epistemic"))
cfg_null = dataclasses.replace(cfg, dynamics=dataclasses.replace(cfg.dynamics, kernel="null"))
cached_run(cfg_epi, seed=SEED, persist=("posts",))
cached_run(cfg_null, seed=SEED, persist=("posts",))

lift = quality_attention_lift(load_run(cfg_epi, SEED), load_run(cfg_null, SEED))
print(f"quality_attention_lift (epistemic minus null): {lift:+.4f}")
"""))

cells.append(md("""### C6 — repulsion as an ablatable switch

Channel 2's negative weights (reply −0.5, report −2.0) are exactly the
repulsive-influence assumption Más & Flache (2013) flag. `repulsion=False`
zeroes them — it does **not** clamp deltas, which would be a different model
wearing an ablation's name. D6 asks whether bipolarization survives without
it.
"""))

cells.append(code("""from discourse_lab.dynamics.drift import social_weights

print("repulsion on: ", social_weights(cfg))
print("repulsion off:", social_weights(dataclasses.replace(
    cfg, dynamics=dataclasses.replace(cfg.dynamics, repulsion=False))))
"""))

cells.append(md("""### C7 — the spiral of silence

`silence_gate` multiplies posting probability by a perceived-climate factor
(perceived through the feed blend of §12, not the global state), moderated by
conviction. It gates *whether you post*, never what you say — which is what
makes false consensus reachable: the expressed stance distribution can end up
less bimodal than the latent one.
"""))

cells.append(code("""from discourse_lab.metrics.polarization import expressed_vs_latent_bimodality
from discourse_lab.population.marginals import empirical_from_editor

cfg_sil = dataclasses.replace(
    cfg_affect,
    dynamics=dataclasses.replace(cfg_affect.dynamics, silence_gate=1.2, kernel="homophily",
                                 drift="none"),
)
batches = [s.retired_posts for s in run_iter(cfg_sil, seed=SEED)
           if s.retired_posts is not None and len(s.retired_posts) > 0]
all_out = concat_post_batches(batches)
expressed = all_out.stance[all_out.kind == "post", 0]

latent = empirical_from_editor(bins=128, support=(-1.0, 1.0),
                               density=axis["marginal"]["density"]).icdf(
    np.random.default_rng(1).random(8000))

res = expressed_vs_latent_bimodality(expressed, latent)
print(f"expressed bimodality {res['expressed_bimodality']:.3f} vs latent "
      f"{res['latent_bimodality']:.3f} -> false-consensus gap "
      f"{res['false_consensus_gap']:+.3f} (0 would mean the gate never bound)")
"""))

cells.append(md("""### C8 — simple vs complex contagion

The Hawkes draw is a *simple* contagion: intensity counts events, not sources.
`reply_model="threshold"` instead makes reply propensity rise super-linearly
in the count of **distinct** engaged in-neighbours (Centola & Macy 2007).
The separating experiment is the crossover with graph structure — long ties
should help hawkes and hurt threshold. If both respond identically to
clustering, `threshold` is not actually complex.
"""))

cells.append(code("""rows = []
for model in ("hawkes", "threshold"):
    cfg_c = dataclasses.replace(
        cfg,
        dynamics=dataclasses.replace(cfg.dynamics, reply_model=model, n_ticks=30,
                                     hawkes_mu_inherit=1.8),
    )
    traj = pl.DataFrame(
        [{"reply_model": model, **s.metrics} for s in run_iter(cfg_c, seed=SEED)])
    rows.append(traj)
replies = pl.concat(rows)
replies.group_by("reply_model").agg(
    pl.col("n_replies").mean().alias("replies/tick"),
    pl.col("n_replies").tail(10).mean().alias("late replies/tick"),
).sort("reply_model")
"""))

cells.append(md("""### C10 — the identifiability harness

Equifinality is the model's central epistemic risk. `separability` asks
whether two conditions are distinguishable **on the metrics the model
reports**, and names the metric that carries the signal — AUC ≈ 0.5 is a
reportable result, not a failure of the harness. `Design` makes the
discriminating experiments runnable and requires each to state its falsifier.
"""))

cells.append(code("""from discourse_lab.experiments.designs import NAMED_DESIGNS, design_names
from discourse_lab.experiments.identify import separability

print("named designs:", design_names())
d = NAMED_DESIGNS["d5_contagion_crossover"]
print("predictions:", d.predictions)
print("falsifier:  ", d.falsifier)

# the harness on synthetic metrics - one metric carries a planted signal
rng_id = np.random.default_rng(0)
a, b = rng_id.normal(0, 1, (20, 4)), rng_id.normal(0, 1, (20, 4))
b[:, 2] += 2.0
report = separability(a, b, ["camp_share", "bubble", "animus_gap", "r_eff"], seed=0)
print(f"synthetic check: AUC {report.auc:.2f}, carried by '{report.top_metric}'")
"""))

nb["cells"].extend(cells)
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("appended", len(cells), "cells ->", len(nb["cells"]), "total")
