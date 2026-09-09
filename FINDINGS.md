# Findings

Measured results and the negative results behind them. Kept out of the demo
notebook so that notebook stays a tour of the API; kept out of the module
docstrings' way by pointing at them rather than repeating them.

Unless stated otherwise: N=1500, 30 ticks, `exposure_sample_rate=0.10`,
10 seeds, each cell against its own matched `kernel="null"` twin.
Bold = resolves against seed-to-seed spread.

## The screen: one lever at a time

16 cells x 10 seeds x 2.

| lever | cross-camp | rank_penalty | feed bubble | top-1% share |
|---|---|---|---|---|
| affinity ranking | **-0.078** | **-0.148** | **-0.099** | -0.002 |
| engagement_optimized | **-0.040** | **-0.068** | **-0.051** | **+0.014** |
| popularity ranking | -0.001 | -0.003 | -0.003 | **+0.026** |
| inject_k 0 -> 20 | **+0.048** | **-0.018** | **+0.039** | **-0.005** |
| long_tie 0.05 -> 0.4 | **+0.042** | +0.002 | **+0.037** | **-0.003** |
| tau_position 6 -> 2 | -0.001 | +0.002 | -0.002 | **+0.012** |
| attention_budget 30 -> 3 | +0.002 | +0.005 | -0.001 | **+0.006** |

Injection and long ties buy cross-cutting exposure; personalised ranking costs
it; popularity ranking is neutral on exposure and the worst lever for
concentrating attention. `epistemic_alignment` is a null result everywhere —
every cell within +/-0.013 at sd ~0.03.

## The crossing: levers against the ranker

26 cells x 10 seeds x 2, `across=RANKER_BACKGROUND`. Effect on cross-camp
exposure, and `swing` = the spread across backgrounds.

| lever | under `affinity` | under `chronological` | swing |
|---|---|---|---|
| tau_position = 2 | **-0.055** | -0.001 | 0.054 |
| inject_k = 20 | +0.011 | **+0.048** | 0.037 |
| tau_position = 15 | **+0.033** | -0.000 | 0.033 |
| attention_budget = 3 | **-0.019** | +0.002 | 0.021 |
| long_tie_fraction = 0.4 | **+0.037** | **+0.042** | **0.005** |

1. **The attention levers are pure interaction.** `tau_position` and
   `attention_budget` do essentially nothing under a chronological feed and are
   substantial under a personalised one. Rank order under `chronological` is
   recency, independent of stance, so truncating the feed removes a random
   slice; under `affinity` rank order *is* stance order, so truncating removes
   precisely the disagreement. **How far people scroll only matters once the
   feed is sorted by agreement.**
2. **Injection is swamped by the ranker it is meant to correct.** `inject_k=20`
   buys +0.048 under chronological and +0.011 under affinity: the personalising
   ranker demotes the injected cross-cutting items back down. "Add diverse
   content to the feed" is weakest exactly where it is most needed.
3. **Network structure is the robust lever.** `long_tie_fraction` is the only
   one with swing ~0. It is the one recommendation here that does not depend on
   what the platform does next.

Attention concentration (`voice_inequality.top1_share`) shows no interaction at
all — every swing <= 0.002 — so those screen rows can be read as-is.

## Negative results worth keeping

**`attention_budget` at 15/30/60 was a dead-zone sweep, not an inert lever.**
Visibility decays as `exp(-r/tau_position)`, which passes 6.51 items on its
own, so the budget cap removes 10% of survivors at b=15, 1.0% at b=30 and
**0.01% at b=60** — the last two are the same platform. Re-ranged to 30/10/3
(binding on 1% / 22% / 63%) it resolves.
Pinned by `tests/test_attention_budget_binds.py`.

**`algorithmic_share` cannot be rescued, and that is a property of the feed.**
Spec 2.5a makes the candidate set `followers(author) u inject(p, k)`, so
injection is the only source of non-follower candidates and the column is NaN
at `inject_k=0` for every ranker. `rank_penalty` replaces it: the pool is fixed
by the graph and by injection, so all a ranker does is order it. Reads +0.001
for `chronological` and -0.002 for `popularity` (neither can see the viewer),
-0.147 for `affinity`.

**Attention concentration is capped by the graph, not the kernel.** The
population's `prominence` is Pareto(2.30) with a max/mean of 303x, but the
latent-space generator flattens in-degree to alpha ~4.7: a user can only be
followed by the ~`knn_k` users whose neighbourhood contains them. Engagement
per post cannot be more skewed than the audience sizes it is drawn over, which
is why attention Gini reads ~0.67-0.72 against the spec's 0.8-0.95 target.
Marked xfail rather than deleted.

**`homophily_beta` does not control homophily.** 0.35 -> 1.5 moves cross-camp
exposure 0.288 -> 0.280, and does about as little for clustering.

**Trait correlations compose additively and neither mechanism knows about the
other.** Asking for `activity x reply_prop = 0.30` with the shipped archetypes
on yields 0.49, because archetypes that shift two traits together already
correlate them. The library warns; `fig_trait_correlations` shows the realised
matrix rather than the requested one.

**The filter-bubble effect is uniform across archetypes.** Per-user `bubble`
sits at ~-0.19 for lurkers, firebrands and institutions alike under affinity
ranking: it is a property of the ranker, not of who you are.

## Scope

The model has no deliberation and no persuasion-by-reason — drift is social
influence only. It speaks to structural preconditions for democratic discourse,
not to deliberative quality.

The screen is a **screen, not the study**: a flat row means flat *at the base
configuration*. Anything that survives it deserves a crossing against
`dynamics.ranker` before it goes in a paper.

## Change-spec waves (C1-C10)

Recorded 2026, implementing discourse-lab-changes.md. Each entry notes what
changed and which old numbers must not be silently compared against new ones.

**The outrage kernel was retuned (C5) - pre-change outrage results are not
comparable.** Out-group attraction moved from an interaction term
(disagree_x_con, a contrarian-minority phenomenon) to the UNCONDITIONAL
outgroup feature (Rathje et al. 2021: out-group terms raised sharing odds
67% - a general-and-large effect). New authoring: reply outgroup 0.9, like
outgroup 0.5, quote 0.7, report 0.6, with disagree_x_con demoted to a
modifier (0.4-0.6). homophily gained a small negative outgroup weight
(-0.3), which is what makes the two kernels separable on cross-camp
engagement share. The C5 identifiability test (classifier AUC between kernel
conditions, experiments/identify.py) has not been run at scale yet; until
it is, separability of outrage vs homophily on the reported metrics is
open.

**quality now draws author-trait-independent by default (C4).**
quality_trait_coupling=0 zeroes the quality row of the expression map, so
Spearman(quality, engagement) under 
ull sits at the noise floor instead of
inheriting the author-trait alignment of the old map. The old behaviour is
coupling=1. epistemic_alignment levels measured before this change are not
comparable; use quality_attention_lift, which requires the matched null
run.

**The exposure sample gained an ttended flag (C2) and the run format
forked (RUN_FORMAT 5).** Old cached runs recompute once. The echo-chamber
index is now computable on exposed vs attended exposures separately; the
difference (selection_filtering.selection_shift) is the Bakshy choice
component. position_only (the default selection model) reproduces the old
attention behaviour exactly.

**Affect, kernel learning, the silence gate, the threshold reply model and
rewiring all default OFF.** Their conformance tests observe each mechanism's
effect (	ests/test_change_spec.py): animus rises under outrage and not
under 
ull (C1); eply_prop rises under outrage reinforcement and not
under 
ull (C3a); conformity narrows gain variance where habituation
does not (C3b); the silence gate's false-consensus gap is strictly negative
in a bimodal population (C7); the threshold model orders reply propensity by
distinct engaged neighbours (C8); rewiring raises stance assortativity
without touching the cached initial graph (C2.2).

**The C9 gate holds on the combination that produces the tail.**
latent_pa + engagement_optimized + andwagon at N=1200 lands attention
Gini inside [0.8, 0.95] AND reciprocity inside [0.2, 0.4] simultaneously -
both stylized rows at once, which the old configuration could not do.

**The C9 combination was recalibrated post-C1-C10, and the 20-seed gate
caught two things a 2-seed check had missed.** First, seed variance: at
	heta_scale social_proof = 1.0 the Gini straddles the 0.95 band top across
seeds (0.939-0.964+ at N=1200, 60 ticks) - a 2-seed check passed on seed
luck, which is exactly the quietly-half-succeeding fit the change spec
warns about. Second, an interaction worth having on the record: with the
shipped drift="full", C3a's behaviour reinforcement compounds attention
concentration under bandwagon and the Gini overshoots to ~0.97 - the
mechanism working as Brady et al. describe and landing outside the band.
The stylized-fact calibration convention is drift="none" (feed dynamics
isolated from trait feedback, like every other stylized test); experiments
quoting stylized-anchored results WITH drift on must re-run the gate at
their settings (	est_c9_*, DLAB_GATE_SEEDS to widen it).

Recalibrated combination: latent_pa + engagement_optimized + andwagon,
mirror_p 0.02 -> 0.05 (reciprocity headroom; the PA overlay depresses it),
	heta_scale social_proof 1.0 -> 0.6 (popularity-force strength; the
population-level group lever C3b introduced). At 20 seeds: Gini 0.82-0.93,
reciprocity 0.252-0.268 - both rows mid-band with margin on both sides.

**D3 was redesigned around the group lever and the budget was measured.**
Sweeping whole kernels cannot find the quality-vs-popularity crossing point
(epistemic has no social_proof to weaken; bandwagon has no quality to crowd
out), so D3 now ladders 	heta_scale social_proof (0 / 0.5 / 1 / 2) on a
bandwagon kernel with quality weighted in via override. Budget, measured at
17 us/user-tick (~1.4 min per run at N=1e4 x 500): 4 ladder levels x
(model + matched null) x 20 seeds = 160 runs ~ 3.8 h sequential, ~30 min at
8 workers. The homophily arm was dropped - it made no competing prediction
there.

**quality_attention_lift now asserts its null is matched** - the two runs'
configs must differ in dynamics.kernel alone, or it raises. With C1 affect
and C2 selection live, differencing against a bare default-config null
also differences those mechanisms out while the number still reads as
being about the kernel.

## The §5.1 attention-Gini band is calibrated at one kernel and does not transfer

Discovered running experiment01abc (A/B/C: sorting vs. selective exposure vs.
micropublics), which manipulates `dynamics.kernel` as its central lever - the
same axis the C9 gate calibration above is anchored to. Gini across every
config actually run:

| config | N | kernel | attention_gini |
|---|---|---|---|
| calibration of record | 10,000 | bandwagon | 0.817-0.851 |
| experiment01abc SHARED (N reduced) | 1,200 | bandwagon | 0.765-0.815 |
| FULL-scale probe, null arm | 10,000 | null | 0.592 |
| World A | 1,200 | outrage | 0.583 |
| World B | 1,200 | homophily | 0.512 |
| World C | 1,200 | civic | 0.417 |

Scale (10,000 -> 1,200) costs roughly 0.05 off Gini; **the kernel costs
roughly 0.22** - an order of magnitude more. The band ([0.8, 0.95]) is only
reachable with `bandwagon` + `theta_scale=(("social_proof", 0.6),)`
specifically, per the C9 finding above ("the C9 gate holds on the
combination that produces the tail"). That was previously read as "this is
the calibration of record, start experiments from it" - correct as far as
it goes, but it has a sharper implication: **no experiment that manipulates
the kernel can inherit §5.1 gate validity, on ANY arm**, including a
matched-null arm run at `kernel="null"` - `null` misses the band by more
than `outrage` does. This is not fixable by choosing which arm to gate on;
it is a property of the calibration protocol itself.

Two ways out, neither implemented yet: (1) per-kernel Gini bands, calibrated
the same way C9 calibrated the bandwagon band, so a kernel-manipulating
experiment has something valid to gate against; or (2) `attention_gini`
leaves `stylized_gate`'s pass/fail set entirely and becomes a reported
diagnostic (a number attached to every run, never a gate criterion) for any
experiment whose lever is the kernel. Until one of those lands, treat every
Gini failure in a kernel-manipulating experiment as uninformative by
construction (already correctly excluded from `NOT_EVIDENCE` sets
downstream) rather than as a substrate problem worth chasing.

## `tie_strength` is a static follow-graph flag, not a repeated-contact accumulator

`discourse_lab/exposure/kernel.py`'s `compute_features` sets
`"tie_strength": is_follower.astype(float)` - recomputed fresh from current
follow status on every exposure, not a running accumulator over repeated
contact. Confirmed by reading the source directly (not inferred from a
statistical proxy) while building experiment01abc's World C, whose §10.4(b)
requirement is exactly a tie-strength-accumulates-over-contact mechanism.

Checked whether this is only a World-C problem: none of the package's
registered kernels (`outrage`, `homophily`, `bandwagon`, `epistemic`, `null`)
put any weight on `tie_strength` in their theta tables - grepped `named_kernel`
output for every registered `kernel_theta` and found no hits. The one place
it carries real weight is experiment01abc's notebook-registered `civic`
kernel, whose theta puts its *dominant* coefficients on it
(`tie_strength`: 1.2 like / 0.8 repost / 1.0 reply / 0.4 quote, exceeding
`affinity` on two of four action types). Under the actual definition of the
feature, this makes `civic` reduce to "engage more with accounts you already
follow" - a static follow-graph-proximity kernel, not Amin's repeated-encounter
mechanism it was written to express. Not a wrong theory to have, but not the
one the θ table's own comments claim, and it should be rewritten once a real
accumulator exists rather than kept as a stand-in.

General point for anyone adding a kernel: `tie_strength` under the current
implementation is informationally identical to `affinity` (both read follow
status), so a theta table that weights both is double-counting one signal
under two names. Grep new theta tables for `tie_strength` before relying on
it meaning anything beyond "is a follower."

## `group_gain`/`conformity` kernel learning amplifies camp imbalance, but only under a kernel that already weights `outgroup`

FULL-scale run of experiment01abc (N=10,000, 500 ticks, 20 seeds) found
World A's `affective_distance` (the symmetry diagnostic, predicted flat in
the pre-registration) statistically significant, though small relative to
the decision metric (`mean_animus` effect 0.00123 vs `affective_distance`
effect 0.0000673, ~5.5%). B and C, run against the same population and
graph substrate, show no such symmetry break.

Root-caused with a one-seed diagnostic (`R14`, FULL scale, persisted
traits, camp-conditional animus at the final snapshot):

- **`rewire` is not the cause.** A with `dynamics.rewire=True` (run of
  record) and the same config with `rewire=False` produce essentially
  identical camp gaps (0.00481 vs 0.00476) — ruled out empirically, not by
  assumption.
- **Camp sizes are exact (5000/5000)** at this seed, so it isn't population
  imbalance from the stance-projection split either.
- The actual mechanism: `dynamics.kernel_learning="group_gain"` with
  `kernel_learning_rule="conformity"` is set in `SHARED` for all three
  worlds, and learns a per-user multiplicative gain on each feature group,
  including `outgroup` (`outgroup`, `outgroup_x_animus`, `ingroup_x_ident`).
  But only `outrage`'s theta table (`kernel.py:130-135`) gives that group
  nonzero, camp-relevant base weight (0.5-0.9 across `like`/`reply`/`quote`/
  `report`); `homophily` weights `outgroup` *negatively* (avoidance, -0.3),
  and `civic` does not use it at all. `group_gain` only has a channel to
  convert an incidental per-user gain imbalance into a camp-level animus
  split where the underlying kernel already routes engagement through that
  feature group — which is `outrage` alone. That is why A shows the effect
  and B/C do not, independent of `rewire`.

This is an emergent interaction between `group_gain`+`conformity` and the
`outrage` kernel's own (camp-symmetric) `outgroup` weighting, not a bug in
either component alone and not an artifact of the metric —
`affective_distance` is exactly the diagnostic built to catch this. Anyone
running a `kernel_learning != "none"` experiment with a kernel that weights
`outgroup`/`outgroup_x_animus` should check `affective_distance` alongside
whatever camp-symmetric prediction the theory makes; the learning tier can
manufacture camp asymmetry the static kernel's authors never intended,
specifically for kernels built around out-group-directed features.

## `sbm_graph`'s reciprocity-by-chance and clustering ratio do not survive a population-size change (reciprocity fixed; clustering ratio still open)

`graph.sbm_homophily=0.03` was calibrated once, at N=1,200/8 blocks/
mean_degree=40, to land both `reciprocity` and `clustering_ratio` in their
§5.1 bands simultaneously (confirmed 20 seeds: reciprocity 0.25-0.26,
clustering_ratio 3.8-3.9). At N=10,000 with the same `sbm_homophily` and the
same `mean_degree`, the FULL run's own gate check failed on reciprocity
(0.114, band [0.2, 0.4]) — the calibration silently stopped holding at a
different population size, and the run proceeded past a printed gate
failure without anyone routing that failure into the verdicts.

Root cause, confirmed with graph-only diagnostics (no simulation, `sbm_graph`
+ `reciprocity`/`clustering_vs_random` directly): `sbm_graph` solves
`p_within` from `target_edges / (same_pairs + h * diff_pairs)`. With
`mean_degree` (and therefore `target_edges`) held fixed while N scales up,
`same_pairs`/`diff_pairs` scale as N², so `p_within` falls roughly
proportional to 1/N. Reciprocity produced purely by chance from independent
directed-edge draws goes as `p_within²`, so an 8x increase in N (1,200 →
10,000) collapses chance-reciprocity by roughly 64x — exactly the
SMOKE-to-FULL drop observed (~0.25 → ~0.11-0.12).

`add_reciprocity`'s `mirror_p` pass (`network/reciprocity.py`) is *not*
broken by this — it mirrors a fraction of already-drawn edges, so its
contribution (`2q/(1+q)`) is density-invariant by construction. It was
simply never turned on for World C (`sbm_mirror_p` defaults to 0.0, unused
at SMOKE because chance alone sufficed there). Turning it on at FULL scale
does restore reciprocity into band — confirmed with a graph-only sweep at
N=10,000 (no simulation): mirror_p 0.00→0.20 takes reciprocity 0.11→0.40
roughly linearly at fixed `sbm_homophily=0.03`. The naive `2q/(1+q)` formula
(the module's own docstring, worked out against a different generator's
baseline chance-reciprocity) is not a reliable predictor here — `q=0.25`
overshoots to 0.456, above the band's 0.4 ceiling. A finer sweep (10 seeds
per value) found `sbm_mirror_p=0.13` centers reciprocity at 0.309±0.001,
comfortable margin on both sides of [0.2, 0.4].

**Applied and confirmed**, not just diagnosed: `sbm_mirror_p=0.13` set on
both World C and `EXTRA_CONTROLS["B_on_C_graph"]` (which builds its own
independent SBM graph at the same homophily; `C_structure` uses the default
`latent_space` graph and was never affected). C's arm was re-run in full at
FULL scale (C, C-null, C_structure, B_on_C_graph — 20 seeds, 80 runs) with
the fix applied: the real `stylized_gate` check now reports World C failing
*only* on attention_gini (expected/universal, see the Gini finding above),
reciprocity failure is gone. Note `clustering_ratio` is not one of
`stylized_gate`'s checked rows (`GATE_ROWS = ("attention_gini",
"reciprocity")` in `experiments/gate.py`) — fixing reciprocity alone clears
the gate that actually blocks the run, independent of whatever
`clustering_ratio` is doing.

That is not the whole fix, though. `clustering_ratio` **also** collapses
with N at fixed `sbm_homophily` and block count (3.8-3.9 at SMOKE → 1.54 at
FULL, `sbm_mirror_p=0`), and — unlike reciprocity — lowering `sbm_homophily`
further does not recover it: it *rises* as h→0 (1.54 at h=0.03 to 2.51 at
h=0.0005) but plateaus below the required ≥3.0 no matter how small h gets.
At N=10,000/8 blocks, mean block size is 1,250 users; the local edge density
that a fixed `mean_degree` and shrinking `p_within` can put inside a block
of that size has a triangle-density ceiling `clustering_vs_random` cannot
clear, independent of how homophilous the generator is told to be.

The lever that does clear it is block *count*, not `sbm_homophily`: raising
`population.n_topics` (which sets the number of `sbm_block_source=
"topic_affinity"` blocks) from 8 to 32 at N=10,000 gives mean block size 313
and clears both bands easily (reciprocity 0.20 even with `sbm_mirror_p=0`,
clustering_ratio 8.77). But `n_topics` is a `PopulationConfig` field shared
by every world built from `SHARED` in experiment01abc — raising it changes
A's and B's topic-affinity trait dimensionality too, not just C's graph, so
it is a design change to the shared population, not a graph-only fix scoped
to World C.

**Net: no `sbm`-generator calibration is population-size-invariant as
currently parameterized.** Anyone running an `sbm`-graph experiment at a
different N than it was calibrated at must re-check both bands, not assume
either survives. `sbm_mirror_p` recovers reciprocity (done here); the
`clustering_ratio` ceiling remains unfixed and needs either more blocks (a
population-wide change, if blocks come from `topic_affinity`) or a generator
change (e.g. block-size-invariant p_within scaling) — but since it isn't a
gated quantity, it does not block reporting a result, only bears on how
literally to read the original SMOKE-scale calibration comment's claim of
"community clustering" in the graph.

With reciprocity fixed, World C's FULL-scale backfire result stands: re-run
at 20 seeds, `mean_animus` effect +0.00237 (vs A's +0.00123), essentially
unchanged from the pre-fix figure (+0.00236) — reciprocity was a second-order
graph property here, not the dominant one driving the effect (that's
`cross_camp_tie_share` ≈0.50, the graph's camp-blindness, which the mirror_p
fix does not touch). The result is now quotable as a claim about a
camp-blind topic-block graph's substrate, subject to the scoping in the
kernel-learning and `tie_strength` findings above.

## Normalizing World C's animus effect by engagement volume weakens the backfire reading, not confirms it

World C's raw `mean_animus` effect (+0.00237) is ~1.9x World A's (+0.00123),
the headline of the backfire finding above. But C's topic-blocked graph also
drives far more raw cascade volume: computed directly from the FULL run's
own persisted seed-0 metrics (`metrics.parquet`, `n_engagements` summed over
all 500 ticks — no re-run needed), C's world arm generates 2,588,398
engagements against its null's 1,066,265 (marginal dose +152.2/user); A's
world arm generates 1,636,103 against its null's 1,153,732 (marginal dose
+48.2/user) — C drives **3.15x A's marginal engagement volume**.

Dividing the animus effect by the marginal engagement dose it took to
produce it (`mean_animus` effect / marginal engagements-per-user, matching
the world-minus-null differencing already used for the animus effect
itself):

| world | marginal dose/user | animus effect | **animus per marginal engagement** |
|---|---|---|---|
| A | 48.2  | +0.001228 | **2.55e-5** |
| B | 15.8  | -0.000144 | -9.1e-6 (noise; B is flat) |
| C | 152.2 | +0.002365 | **1.55e-5** |

Per contact, C is **61% as hostility-inducing as A, not more**. The raw
comparison (C's animus effect ~1.9x A's) is substantially a cascade-volume
artifact: C's camp-blind graph produces far more total contact, and that
larger volume — at a *lower* per-contact rate — is what the aggregate number
reflects. This does not erase the backfire finding (C's population still
ends up more hostile than A's, in aggregate, than the theory predicted), but
it changes what mechanism is licensed: "C's structure generates more total
out-group contact, and more contact costs animus even at a below-A per-unit
rate" is a different and weaker claim than "C's mechanism is a more potent
per-contact hostility generator than A's." Only seed 0 is available for this
calculation (traits/metrics for other seeds were deleted per-seed to survive
the disk-space crisis), so there is no cross-seed CI — but the gap (1.6x) is
far larger than the seed-to-seed noise visible elsewhere in this run.

## D2 (kernel identifiability): re-run confirmed with the volume-artifact family excluded, and it's genuinely separable

Three review rounds excluded volume artifacts from experiment01abc's D2
kernel-pair separability test one name at a time -- `n_posts`, then
`n_engagements`, then the whole `n_*` family -- and each time the AUC=1.00
carrier just moved to the next uncaught count. The actual FULL run of
record's own D2 output (`experiment01abc_FULL_out.ipynb`) predates even the
`n_*`-family fix and still shows `n_replies` carrying AUC=1.00; this was
confirmed by reading that file directly, not assumed, before re-running
anything.

Re-running D2's exact code (population/graph/dynamics config and gate
results checked identical to the run of record) with the `n_*` family
excluded still showed AUC=1.00, carried by `open_threads` -- `float(len(
self.threads))` in `dynamics/tick.py`, a raw open-thread count that simply
doesn't start with `n_`. The real exclusion criterion was never "starts
with n_"; it's "is this an event/entity COUNT" vs "is this a rate, index,
or learned-parameter deviation". Excluding `open_threads` too, a further
re-run (20 seeds, all three kernel pairs) still reports AUC=1.00, now
carried by `kernel_gain_dev` (mean `|learned per-user gain - 1|` under
`group_gain`/`conformity` -- a direct behavioral signature of how hard each
kernel's own theta table drives adaptation) and `reply_fallback_rate` (a
proportion, not a count).

This is the honest result, not a bug to keep chasing: the three kernels'
aggregate behavioral signatures are genuinely, perfectly separable via
mechanism at N=10,000/20 seeds, once every volume artifact is actually
excluded. It is not a non-identification finding -- but it does mean D2, as
built, cannot distinguish "these kernels differ enormously" from "these
kernels differ just enough to be theoretically distinct," since AUC
saturates at 1.00 either way at this population size and seed count. Full
per-pair carriers and kept-metric lists: `results/abc/d2_identifiability.json`.
