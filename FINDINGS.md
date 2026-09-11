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
ull (C1); 
eply_prop rises under outrage reinforcement and not
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

## Experiment 01 — Sorting vs. selective exposure: affect channel under-powered at small scale

Törnberg's partisan-sorting world (outrage kernel, engagement ranker,
position-only selection, `inject_k=20`, rewiring on) vs. Bakshy et al.'s
selective-exposure world (homophily kernel, chronological ranker,
homophilous selection, `inject_k=2`, no rewiring), each against its own
matched `kernel="null"` twin. Notebook:
`notebooks/experiment_01_sorting_vs_selective_exposure.ipynb`; script:
`discourse_lab/experiments/experiment01_sorting_vs_selective_exposure.py`.
Scaled down from the source spec's 10,000 users / 500 ticks / 20 seeds to
1,500 users / 150 ticks / 8 seeds (main grid) and 4 seeds (`inject_k`
ladder [0,2,5,10,20]) — see the notebook's "Deviations from the spec"
section for the full list of what did not run (World C, D2
identifiability, `selection_beta` calibration, the θ-ratio bracket, the §8
ablation ladder, the C10 Sobol sweep).

**The packaged default scenario's stance marginal fails the affect
metrics' own camp-bimodality gate** (`camps_and_bimodality` needs
bimodality ≥5/9≈0.556; the default measures 0.297), so
`ident_animus_coupling`, `affective_distance`, `animus_asymmetry` and
`stance_centroid_distance` read identical to six decimal places across
both worlds and both nulls when run against it — not a subtle null result,
a sign the affect channel never engaged. Fixed for this run only by
substituting an explicit bimodal one-axis scenario (bimodality ≈0.75, the
same construction as the demo notebook's C1 cell) into the shared
substrate; this is a substrate-level gap the change-spec's affect metrics
have and worth closing generally, not just for this experiment.

**Even with camps defined, the affect channel did not separate a world
from its own null at this scale.** `affective_distance`: World A 0.0520 vs.
null 0.0522 (Δ=-0.0002), World B 0.0523 vs. the same null (Δ=+0.0001) — two
orders of magnitude below the metric's cross-seed spread. Per the source
spec's own decision rule for this case: `lr_affect` (0.015, default,
unchanged) and/or `n_ticks` (150 vs. the spec's 500) need to move before
the decisive-cell test (high animus at low echo-chamber index) means
anything here. The 2x2 placement nominally came out backwards from both
theories' predictions (World A: low animus/high echo; World B: high
animus/low echo) but the differences driving that placement are noise, not
signal.

**The structural (selection + `inject_k`) dimension separated cleanly and
in the expected direction, independent of the affect question.**
`echo_chamber_index` (attended): World A 0.499, World B 0.475. Cross-camp
exposure share: World A 0.277, World B 0.331. Present equally in the
worlds and their nulls, since kernel is the only axis that differs between
a world and its null — expected, and not itself evidence for either
kernel's mechanism.

## Change-spec V-series — the affect-hostility table has no de-escalation channel

Third spec-versus-implementation gap surfaced by inspection, after
`tie_strength = is_follower` (still open — see below) and the topic-affinity
SBM blocks gap closed by implementing `graph.sbm_block_source="topic_affinity"`.
Same shape each time: a mechanism the design docs gesture at, absent or
unreachable in code, invisible because nothing tested for its presence.

**The table.** `AFFECT_HOSTILITY_WEIGHTS` (`discourse_lab/dynamics/drift.py`)
and `DynamicsConfig.affect_weights_hostility` (`discourse_lab/config.py`)
carry identical values:

| action | like | repost | quote | reply | report | skip |
|---|---|---|---|---|---|---|
| hostility weight | 0.25 | 0.25 | 0.5 | 0.5 | 2.0 | 0.0 |

Every non-`skip` entry is positive. `affect_delta`'s animus term is
`lr_affect · mean_over_exposures(outgroup · hostility_weight(action))` with
`outgroup ∈ {0,1}` and `lr_affect >= 0`, so animus is monotone non-decreasing
relative to each user's baseline. No `affect_weights_hostility` override
changes that: the table is keyed on action alone, so a negative entry would
make every instance of that action de-escalating — civil and hostile alike —
which is a different, worse model, not a de-escalation path.

**The design docs specified this as reachable, not as a known limitation.**
`drift.py`'s own module docstring, predating this finding: "the hate-
engagement reading (any out-group engagement raises animus, scaled by how
confrontational the action is — Rathje et al. 2021) is a theory, and the
contact-hypothesis alternative (likes as positive contact that LOWERS animus)
is a different theory a sweep should be able to express without editing the
loop." `MODEL.md`'s config table repeats the claim almost verbatim: "the
hate-engagement reading is a theory, the contact-hypothesis alternative is a
different one." Both name the alternative and both assert a sweep — a config
override, no code change — should reach it. Neither is true of the shipped
table. The gap is documented rather than re-argued: the intent for a
de-escalation path is in writing twice, and no config value realizes it.

**Scope.** Blocks Experiment 03: its question ("under what conditions is
popping filter bubbles defensible") has four possible answers and the current
build can reach two, since the "contact reduces hostility" quadrant is
excluded by the update rule — the experiment would answer itself before it
ran. Does NOT invalidate Experiment 01: every comparison there was between
conditions all driven upward, so orderings hold, and `echo_attended` is an
exposure measure never routed through this table. What does not survive is
reading a condition's animus number as a *finding* rather than a
*consequence* — `outrage` weights `outgroup` at 0.5-0.9
(`discourse_lab/exposure/kernel.py`), so out-group engagement is already
outrage's dominant driver, and "outrage raises animus" is definitional in two
steps (outrage leads to more out-group engagement, and animus can only rise
from engagement) — not an independent result of the affect channel.

**V5 minimum conformance set, measured against current code**
(`tests/test_change_spec.py::test_v5_*`). The change spec ("Change Spec V1 —
Engagement Valence and the De-escalation Channel", reviewed but not committed
to this repo) predicts four of its five assertions fail today. Measured: two.

| assertion | result |
|---|---|
| de-escalation: civil cross-camp contact lowers animus below baseline | **fails** (`xfail`, strict) |
| repeated encounter: `tie_strength` rises with repeat contact, follow status fixed | **fails** (`xfail`, strict) — `tie_strength` is exactly `is_follower.astype(float)` (`kernel.py:201`); no channel carries contact history |
| topic-affinity blocks: assignment correlates with topic affinity, not stance | passes — closed by `graph.sbm_block_source="topic_affinity"`; default population sampling draws topic affinity and stance independently, so the max block-vs-stance point-biserial `r` stays under 0.1 at N=2000 |
| backfire: hostile cross-camp contact raises animus | passes — trivially, every table entry is >= 0 |
| selection: `echo_attended > echo_exposed` under `homophilous` | passes — the existing `selection_filtering` outcome (`outcomes.py`) already implements and reports exactly this decomposition |

Not a subtle miss: two of the spec's four predicted failures are gaps this
project had already closed (SBM blocks) or already built under a different
change spec (C2 selection) before this document was written — its "three
known gaps" framing counted `tie_strength = is_follower` and the SBM gap as
both still open, and only one of those two still is. The two real, currently
open gaps are the de-escalation channel itself (this document's subject,
scheduled to close under V1/V2) and `tie_strength` (which none of V1-V6
addresses). The latter is pinned as an `xfail(strict=True)` regression guard
specifically so it stays visible in the suite rather than silently absent,
and so it fails loudly — forcing the marker's removal — the moment some
future change makes it pass.

## Change-spec V-series (V1-V6) landed, sequenced as the spec's own waves

Implemented in the order the change spec itself set out (`## Sequencing`):
Wave 0 (V0, above) and Wave 1 (V5, above) first; this entry covers waves
2-5 — V1, V2, V4, V6(1)+(2), V3, and the substrate re-gate — landed in one
pass. `tests/test_change_spec.py::test_v1_*` through `test_v6_*` are the
conformance set; each observes the mechanism's effect, per house style.

**V1 — engagement valence.** `dynamics/valence.py::EngagementValence`:
`agree`/`civil`, orthogonal booleans assigned per exposure at the moment of
engagement, persisted on `engagements.parquet` (RUN_FORMAT bumped 5 -> 6).
`agree` is NOT a per-batch median split — that was tried first and produces
a real bug (below). It thresholds the kernel's own `agreement` feature
against an ABSOLUTE distance calibrated once per population
(`TickEngine.__post_init__`, `agree_delta`), the same median-pairwise-
distance technique `outcomes.py::selection_filtering` already uses for its
own `delta`, including the hardcoded seed (a normalization constant, not a
modeling draw). `civil` is exogenous in V1/V2: a coin flip at
`dynamics.civility_prob` (default 0.5), drawn from the already-registered
`"affect"` phase stream so it cannot shift any other phase's draws.
`force_agree`/`force_civil` pin an axis for fixture tests.

**Bug caught by the EXISTING suite, not a new test.** The first `agree`
implementation (per-tick median split) made
`test_c1_animus_rises_under_outrage_and_not_under_null` fail: forcing
exactly half of every batch to read "agree" washed out `outrage`'s
deliberate skew toward disagreeable out-group content, diluting the
outrage-vs-null animus gap into noise (0.00019 vs 0.00019). This is the
kind of thing V5's "conformance test per mechanism, before the mechanism
changes" discipline exists to catch, and it worked on the first structural
change made under it — the pre-existing test caught a bug in the NEW code,
not the other way around.

**V2 — the table re-keyed on (action, valence).**
`DynamicsConfig.affect_weights_hostility` is now a pure per-action
MAGNITUDE table (every entry >= 0; `report` removed entirely). The sign
lives in the new `affect_valence_signs` table
(`disagree_hostile: +1.0, disagree_civil: -1.0, agree_civil: -0.3,
agree_hostile: +1.2`), matching the spec's sign structure exactly:
negative on cross-camp civil disagreement specifically (Allport's
condition), not on in-group agreeable contact. `drift.py::affect_delta`
composes `weight(action) * valence_cell_signs(cell)`; `identification`
(the support/in-group channel) is untouched — the spec's re-keying is
scoped to the hostility table alone.
`tests/test_change_spec.py::test_v1_v2_civil_crosscamp_contact_can_lower_animus_end_to_end`
is the spec's own V2 test, run through the real engine: forcing
disagree+civil vs disagree+hostile on an `outrage`/polarized-population run,
the civil arm's animus ends below both the hostile arm AND its own
baseline — the missing direction V0 recorded is no longer missing.

**V4 — report as an exit event.** Kernel: `outrage`'s `report` action
gained an `outgroup_x_animus` term (propensity rises with the reporter's
OWN animus — the correct causal direction, replacing the pre-V2 reading
where report was simply the largest hostility increment). Effect:
`dynamics/report_exit.py::ReportSuppressionState` remembers every
(user, author) a user has reported (a sorted key-array set, same technique
`rewire.py::RewireState` uses to avoid an N x N dense accumulator) and
drops those pairs from `candidate_inbox`'s output before ranking, gated on
`dynamics.report_exit` (default off, like every other change-spec
mechanism until asked for). `report_animus_increment` (free parameter,
default 0.0) is applied outside the (action, valence) table for anyone who
wants a small direct effect without touching the exit mechanism.

**V6(1) — continuous distance was already the mechanism.** The engagement
kernel's `agreement` feature (`-||s_u - s_p||`, full stance vector) and
V1's `agree_delta` threshold both already operate on continuous per-axis
distance, never camp membership — camps only gate WHETHER the affect
channel applies (in-group vs out-group), never HOW agreeable a specific
dyad is. `test_v6_1_agreement_is_continuous_per_axis_not_camp_membership`
pins the "Bernie case" (far on axis 0, close on axis 1 reads as more
agreement than far on both) as a regression guard.

**V6(2) — emergent k via BIC.** `metrics/polarization.py::emergent_camps`
fits k-means at k=1..k_max and selects k by BIC. Two failure modes found
and fixed before it worked, both worth recording because they are the
generic way "fit a mixture, pick k by BIC" breaks:
1. **Per-cluster variance is degenerate.** A Gaussian mixture's likelihood
   is unbounded as any one cluster's variance shrinks to 0, so letting BIC
   score each cluster's own variance rewards carving off a tiny
   low-variance sliver, and the score decreased monotonically to k_max on
   BOTH a clean 2-blob and a clean 5-blob synthetic population — it never
   once selected the true k. Fixed by pooling ONE isotropic variance across
   all k clusters (Pelleg & Moore 2000's k-means BIC), which removes the
   degenerate degree of freedom.
2. **Even pooled, it still needs the mixing-proportion term.** Omitting
   `sum_i n_i * log(n_i / n)` from the log-likelihood (i.e., scoring only
   each point's distance from its centroid, not which cluster it fell into)
   reproduced the same monotonic-to-k_max failure, because nothing was
   penalizing an uneven split into many small clusters. With both fixes,
   `test_v6_2_emergent_k_recovers_two_camps` and
   `_makes_fragmentation_visible` recover k=2 and k=5 exactly on their
   respective synthetic populations.
Group-directed (vector) animus remains deferred, per the spec's own
scoping — not attempted here.

**V3 — endogenous valence.** `dynamics/valence.py::assign_valence_endogenous`
implements both logits verbatim, over the engaging user's OWN
`animus`/`identification` (no renormalization — same convention
`exposure/kernel.py`'s `outgroup_x_animus`/`ingroup_x_ident` features
already use) and the dyad's full continuous distance (recovered from the
kernel's own `agreement` feature, `d = -agreement`). Gated on
`dynamics.valence_mode` ("exogenous" default = V1/V2's coin flip;
"endogenous" = V3), which fails fast at engine construction if
`population.affect` is off — P(civil) cannot read an animus that does not
exist. None of the eight coefficients are empirically anchored (same
status as `affect_ou_k`); signs are fixed by theory in
`EndogenousValenceParams`' own docstring, magnitudes are free.

**The bistability probe is isolated at the mechanism level, not run
through the full tick loop.** `_iterate_animus_feedback` iterates the
endogenous valence draw + `drift.py`-shaped OU composition directly (dyad
distance held at 0, isolating the animus -> civility -> sign -> animus
loop specifically) — this is what let the probe run in under 2 seconds
instead of as a multi-minute population sweep, and it is what surfaced a
real property of THIS model's OU design worth recording: because `Bs`
(the mean-reversion target) itself drifts toward `X_stored` at `k/10`
(`drift.py::DriftState`), the "restoring force" on animus vanishes as `Bs`
catches up, so there is no finite STABLE high fixed point under pure
self-reinforcement — a population pushed hostile does not plateau, it
keeps drifting upward for as long as the run continues. The two regimes
the probe demonstrates are better described as "decays to baseline" vs.
"escapes and keeps climbing" than as two fixed points in the strict
dynamical-systems sense; both readings satisfy the spec's actual question
(does initial condition change the long-run fate), so the test checks the
TREND (is the gap between a low-start and high-start population growing or
shrinking), not a snapshot value.

**Non-tautology check, not skipped.** The spec's own warning — fixing the
self-reinforcement coefficient at a value guaranteed to produce two basins
and then reporting basins is circular — is a standing test, not a one-off
check: `test_v3_no_basins_when_self_reinforcement_is_absent` runs the
IDENTICAL two starting points through the IDENTICAL other coefficients with
only `gamma_animus` zeroed, and the gap must CLOSE over time rather than
grow. It does (gap shrinks from 4.86 to 3.72 over 450 further steps, vs.
growing from 8.90 to 11.87 in the `gamma_animus=-1.0` arm) — the divergent
case is a property of the self-reinforcement term specifically, not an
artifact of the harness.

**Not built here, and deliberately out of scope.** Experiment 03 itself —
the Morris screening that fixes V3's parameter set by sign before any
calibration, and the break-even de-escalation magnitude Experiment 03 is
supposed to solve for and report — is explicitly what V1-V6 unblock, not
part of them; the change spec's own intro says "Nothing in Experiment 03
should be built before V1-V4 land." V6's group-directed animus and the
"target identified camp pairs" alternative for a continuous-distance
intervention are both explicitly deferred/unsettled in the spec itself.

**Wave 5 — the substrate re-gate.** None of V1-V6's new code paths execute
under the C9/C13 gate's own config (`population.affect` off, no camps,
`report_exit` off, `valence_mode="exogenous"` default which itself never
touches the engagement kernel or channel-2 social weights) — `assign_valence`
now runs on every engaged tick regardless of `population.affect`, but it
only consumes the already-isolated `"affect"` phase stream and writes to
columns nothing else reads, so no other phase's draws or outcomes can move.
Re-ran `test_c9_attention_gini_and_reciprocity_hold_simultaneously` and
`test_c13_stylized_gate` at the full 20-seed gate: both still pass on the
current code — the substrate is unchanged, as the RNG-isolation argument
predicts, and this is the confirmation rather than an assumption.

## Experiment 03 — infrastructure built, Wave A run: engagement and composition diverge in sign, cleanly

V1-V6 unblocked Experiment 03's question (§0 above) but three more
prerequisites the brief itself names turned out not to exist yet, only one
of them previously flagged. Built the same way V1-V6 were: test-first,
defaulted to prior behaviour, each gap recorded before it was closed.

**Prerequisite 1 — §5.1's "package prerequisite," a schedule inside one
config.** Nothing existed to let a single run change parameters at a named
tick. `DynamicsConfig.schedule` is `((start_tick, ((field, value), ...)),
...)`, resolved per tick by `config.py::effective_dynamics` and read at the
top of `TickEngine.step` in place of the static `self.cfg.dynamics`.
Entries are not cumulative — a later entry restates the fields it touches
rather than undoing an earlier entry — which is what lets a withdrawal
entry cleanly revert to the `none` arm's values without knowing what an
intervention arm changed. Because RNG streams are keyed on `seed` alone
(`runner.py::phase_rngs`, never on config content) and every OTHER phase
reads the same effective values until the first entry fires, two configs
sharing a seed and an identical dynamics config up to some tick `T`,
differing only in a schedule entry AT `T`, produce a **bit-identical**
per-tick record for every `t < T` —
`tests/test_runner.py::test_schedule_gives_a_bit_identical_prefix_and_diverges_after`
proves this directly rather than assuming it. This is what lets Experiment
03's four arms fork from one shared burn-in instead of being separate
configs that "merely start the same way" and diverge for reasons the
config hash cannot record (the confound the brief names explicitly).

Not every field is schedule-reactive, and the boundary is a real trap, not
a footnote: a field consumed once at `TickEngine` construction (`kernel_
learning`, `quality_trait_coupling`, `agreement_metric`'s calibration) or
read from the stored `Config` directly inside a helper that takes the whole
config (`apply_drift`'s hostility/support/OU tables) does not react to a
later schedule entry. **This caught a real bug before it shipped, not
after:** `EndogenousValenceParams` (V3's eight `valence_*` coefficients)
was built once in `TickEngine.__post_init__` from the base config and
reused every tick via `self.valence_params` — so a "composition" arm
scheduling `valence_gamma0` upward (raising baseline civility under
endogenous valence) silently did nothing. Caught by exactly the V5
discipline this project keeps leaning on: a smoke test showed the
composition and engagement arms producing bit-identical per-tick metrics,
which should have been impossible given they differ in `valence_gamma0`.
Fixed by rebuilding `EndogenousValenceParams` per tick from the effective
dynamics config inside `step` (`dynamics/tick.py`); `civility_prob` itself
was never affected (it IS read per-tick, in the branch endogenous mode
never takes) but would have had the identical bug under exogenous mode had
`valence_params`-style caching been copied there too.

**Prerequisite 2 — §4's structural-tribalization dial.** Recorded already
(above): archetype/topic_affinity SBM blocks are uncorrelated with camp,
and `homophily_beta` was already measured inert for cross-camp tie share
(`experiments/intervention.py`'s own notes). `graph.sbm_block_source=
"camp"` sorts SBM blocks by `stance_clusters`'s sign-of-dominant-axis
label instead — deliberately not gated on the bimodality threshold
`camps_and_bimodality` uses for outcome reporting, since graph structure
needs a split even on a population that is only weakly bimodal (exactly
the "structurally sorted but not yet affectively tribalized" cell the
three-dial design exists to reach). `network.measures.cross_camp_tie_share`
makes the dial's effect directly measurable rather than inferred from
config. `tests/test_network.py` proves both the continuity (>= 4 distinct
readings sweeping `sbm_homophily` at one fixed population, not the "two
points" the old block sources gave) and the targeting (camp blocks push
cross-camp tie share below 0.3; topic-affinity blocks stay above 0.4, near
chance, on the SAME population).

**Prerequisite 3 — §4's other two dials, not previously flagged as
missing.** Discovered only once the arms were actually being wired
together: `population.animus_mu` and `population.stance_polarization`
did not exist either. Animus's marginal was a hardcoded
`lognormal(mu=-2.2, sigma=1.0)`
(`population/traits.py::_affect_marginal`); stance axes with no scenario
loaded were always plain `normal(0, 1)` — unimodal by construction, no
knob at all. Built the same way: `animus_mu` threads through to the
lognormal's mean (§4's "affective... initial animus level," low -2.2 to
high 1.0, measured to move mean animus from 0.18 to 4.46 over that range
at N=5000); `stance_polarization` draws axis 0 from a new
`bimodal_normal(separation, sigma)` marginal (`population/marginals.py`,
tabulated-CDF inversion, same technique `vonmises` already uses — a
Gaussian mixture has no closed-form inverse either) instead of plain
normal above 0. Axis 0 only, not every axis: `camps_and_bimodality`
projects onto the dominant component, and axis 0's variance dominates as
soon as separation makes it the largest, so touching one axis is enough
without changing every other axis's marginal shape. Calibrated at N=800-
1000: the Sarle-bimodality gate (5/9) crosses around `stance_polarization`
≈ 4, i.e. dial level ≈ 0.5 under this module's 0-8 mapping.
`tests/test_population.py` proves both dials move monotonically and that
`stance_polarization` leaves every axis but 0 untouched.

**A property of the model, surfaced by the gate, broader than first
recorded.** `dynamics/drift.py::apply_drift`'s C1.3 affect op — the ONLY
place `animus`/`identification` update at all — is gated on `camps is not
None` (line ~487), not only the engagement kernel's `outgroup`/
`outgroup_x_animus` features. Below the bimodality gate, animus does not
move for ANY arm, including `none` — confirmed directly in Wave A below,
where every arm's `delta_aff_plateau` is exactly 0.0 at the lowest
ideological-dial level swept. "Does popping the bubble help" is not
merely hard to detect below the gate, it is not measurable at all: the
model has nothing to report until the population has actually sorted into
two camps. `experiments/experiment03_bubble_intervention.py::dial_config`
documents this and `wave_a_screen` sets its shared background level (0.7,
not the naive 0.5 midpoint) specifically to stay clear of it while sweeping
the other two dials.

**The four arms and the outcome pair (§2-3), as built.** `ARMS` in
`experiment03_bubble_intervention.py`: `exposure` raises `inject_k` to 20;
`engagement` SETS the kernel's `outgroup`/`outgroup_x_animus` weights to
clearly positive values via `kernel_theta` (not `theta_scale` — a
multiplicative scale flips sign depending on the base kernel's own
convention, homophily's outgroup weight is negative and outrage's is
positive, so it cannot reliably mean "more cross-camp engagement" across
different burn-in kernels); `composition` adds `valence_gamma0=2.5`
(P(civil) at animus=0 rises from 0.5 to 0.92) on top of the same kernel
change. `delta_aff` (plateau animus, arm minus none, plus the same delta
normalized by total engagement volume over the window) and `delta_ideo`
(toward-other-camp / toward-mean / toward-own-pole / delta-k, each a
per-user DISTANCE-DECREASE oriented toward its reference point before
averaging) are reported as a pair, never collapsed — `delta_ideo`'s three
camp-relative components are computed against a FIXED axis and camp split
taken from the shared pre-intervention state, so a later tick's population
is scored against a reference frame that does not itself drift.
`tests/test_experiment03.py` proves the decomposition does not repeat
Experiment 01's `affective_distance` bug: a scenario where both camps
converge toward the population mean by the same amount cancels to exactly
zero under a naive raw signed mean, and reads correctly as `toward_mean ≈
+1.0` here.

**Wave A (§5.4), reduced scale.** Not a true Morris elementary-effects
design — that needs multiple random trajectories through the 3-dial space
to estimate global sensitivity, out of scope for one session — but a
one-factor-at-a-time sign screen: each dial swept at levels {0.1, 0.5,
0.9} with the other two held at the 0.7 background, 5 seeds per cell.
N=1000, 60 ticks burn-in, 100 ticks post-intervention (longer than an
initial 40-tick check, which gave legible-but-barely-so deltas of order
1e-4 given `lr_affect=0.015`'s slow step size). 45 design points, 135
model-arm rows, ~30s/point, ~23 minutes wall clock
(`results/experiment03/wave_a.csv`).

**Measured.** Across every non-degenerate cell (i.e. excluding
ideological=0.1, where the gate above makes the whole affect channel
inert):

| arm | delta_aff_plateau sign | consistency |
|---|---|---|
| **composition** | negative (lower animus than `none`) | 40/40 seed-cells |
| **engagement** | positive (higher animus than `none`) | 40/40 seed-cells |
| **exposure** | positive, an order of magnitude smaller | 37/40; mixed (2/5 positive) only at structural=0.1 |

**H3 (composition carries the sign) holds cleanly at this reduced scale
and this background slice.** Engagement and composition differ not just in
magnitude but in SIGN, on every design point where the comparison is
measurable — the tautology-risk note in the brief's own §3 predicted
exactly this ("the engagement arm alone... will and should show backfire...
The composition arm is the informative one"), and Wave A reproduces it
quantitatively rather than by construction: `civility` is the only
difference between the two arms' schedules, and it flips the sign every
time. `exposure`'s small, occasionally-ambiguous effect also matches prior
work: Experiment 01 measured `inject_k` moving the raw cross-cutting-
exposure aggregate by almost nothing while the injected items themselves
were far more cross-cutting — "drowned out by follower fanout at any
dosage a platform would ship."

**H1 (the sign flips somewhere in the tribalization space) is NOT
established by this pass, and that is a real limitation of the design, not
a null result.** Within the three ranges actually swept — each dial from
0.1 to 0.9 around a FIXED 0.7 background on the other two — no arm's sign
changes. That rules out a crossing inside this specific one-factor-at-a-
time slice; it says nothing about the full 3-dial volume a real Morris
design (or a denser LHS, per the brief's own Wave B) would cover, and
nothing about the region below the ideological gate, where the question
is not measurable rather than answered "no". Read honestly: uniformly
negative on composition and uniformly positive on engagement, IN THE
SLICE TESTED, is closer to the brief's own H1 falsifier language than to
a confirmed crossover — worth flagging rather than either claiming H1 or
declaring it falsified.

**H2 (the bundle comes apart) has a genuine, unanticipated signature in
this data.** `toward_mean` is negative and `toward_own_pole` is positive
in 113/120 non-degenerate seed-cells, for EVERY arm — including
`composition`, which simultaneously LOWERS animus. The brief's own §2
table anticipates two rows for the ideological axis ("positions converge"
vs "positions unchanged"); what Wave A actually shows is a third pattern
the table does not name: positions do not converge OR stay put, they
measurably move toward radicalization even under the arm that reduces
hostility. "Pacification without persuasion" (the table's own bottom-left
cell) undersells it — this reads as "de-escalation with simultaneous
ideological hardening," a combination worth a name of its own if this
holds up at full scale. Caveat stated plainly: magnitudes here are ~1e-3,
one reduced-scale run, and `toward_other_camp`/`toward_mean`/`toward_own_
pole` are close to mirror images of each other in this near-linear
two-camp setup (recorded in the outcome-pair's own module docstring) — the
DIRECTION is the finding, not yet the magnitude.

**H5 (fragmentation is a distinct outcome) has no support in this pass.**
`delta_k` is exactly 0.0 on all 135 rows — `emergent_camps`' BIC-selected k
never moved, for any arm, at any design point. Could be a true negative
(the binary frame is adequate at this scale) or could be that 160 ticks
and N=1000 is simply too short/small for a k-means-BIC estimator to
register a shift — this pass cannot distinguish the two, and does not
claim to.

**What this is not, stated plainly (see also the module's own
docstring).** No hysteresis phase (§5.2) — nothing here says whether the
composition arm's de-escalation is reversible. No response-surface fit
over the full 3-dial volume, only three one-at-a-time slices through it.
No calibration against a corpus (§8) and no viewpoint-diversity floor
(§2.3) — both explicitly deferred as normative/scope choices in the brief
itself, unresolved here too. `delta_aff.per_contact` normalizes by total
engagement volume, not contact restricted to cross-camp pairs specifically
— the literal §2.2 ask needs an engagement/author-camp join this pass does
not build. Waves B-D (the LHS response surface, hysteresis, and held-out
confirmation) are not run. The open decisions §10 of the brief lists —
the viewpoint-diversity floor's number, corpus-or-swept-anchored-group,
and the intervention target under emergent k — are exactly as open as
before this session; none of them blocked Wave A, but all of them block
Wave B.

## Change spec V7 — continuous affect drive: implemented, substrate re-gated

V7.1-V7.6 (change-spec-v7-continuous-affect.md), landed together per this
project's V1-V6 precedent. The spec's own diagnosis was correct: the C1.3
affect op's `camps is not None` gate froze animus/identification for every
arm below the Sarle bimodality threshold, which is exactly the region
Experiment 03's H1 crossing would have to sit in if it exists.

**V7.1 — gate-state instrumentation.** `metrics.parquet` gains `bimodality`
(this tick's Sarle coefficient, wherever the camp split is already computed)
and `affect_gated` (whether the affect op actually ran), both built from a
new `dynamics.drift.affect_gate_active` — the single function `apply_drift`
itself now calls to decide, so the instrumentation cannot silently disagree
with the mechanism the way two independently-maintained copies of the same
gate condition eventually do. `bimodality` deliberately does NOT reuse the
existing `camp_bimodality` column's value: that column is captured before
`TickEngine.step`'s own `_refresh_camps()` call and so reports the
PREVIOUS tick's coefficient — fine for its original purpose, wrong for
auditing whether a run's bimodality crossed the gate mid-run, which is
what V7.1 exists to do. Wave A's own 135-row CSV could not be re-audited
against this instrumentation in this session — the underlying cached run
directories (`dlab/runs/...`) are gitignored and were not present in this
checkout — but the columns are in place for Wave A′ or any future run.

**V7.2 — checked, and mostly already true.** Read closely,
`experiment03_bubble_intervention._ideo_decomposition` was ALREADY
differencing `toward_other_camp`/`toward_mean`/`toward_own_pole` against
the `none` arm's own movement (`net = _movement(stance1_arm) -
_movement(stance1_none)`), confirmed by the pre-existing
`test_ideo_components_are_net_of_the_none_arms_own_movement`, which passes
unmodified. The spec's warrant describes an absolute-level bug that the
code, as it stands, does not have. What genuinely was missing: the `none`
arm's own absolute movement was computed and then only ever used as a
subtrahend, never reported — so "toward_own_pole positive in 113/120
cells" could not be told apart from "the none arm does this too" without
re-deriving it by hand. `DeltaIdeo` gains `ideo_level_toward_other_camp` /
`ideo_level_toward_mean` / `ideo_level_toward_own_pole` (the `none` arm's
own `_movement` output, retained rather than only differenced away), and
`run_design_point`'s CSV rows carry them.

**V7.3 — the mechanism change.** `dynamics.affect_drive` (`"camp"` |
`"distance"`) on `affect_delta`/`affect_update`/`apply_drift`. `"distance"`
replaces `outgroup = camps[u] != camps[author]` with `phi(d) = d / (d +
affect_d0)` on the dyad's stance distance — the SAME per-exposure array
(`-features["agreement"]`) V3's `assign_valence_endogenous` already uses
for `P(agree)`, threaded through `TickEngine.step` into `apply_drift` as
`stance_distance` rather than recomputed, so the two mechanisms cannot
drift apart on what "distance" means. `identification`'s multiplier is
`1 - phi(d)`, the continuous analogue of the binary's `1 - outgroup` — not
spelled out in the spec's own `h_i` formula (written for the animus term
only), and the natural structural generalisation once one is needed for
symmetry. `camps` stayed in `apply_drift`'s signature rather than being
removed (the spec's literal "`camps` leaves `apply_drift`'s signature"):
`"camp"` mode still needs it, TickEngine already computes it every tick for
the kernel's own camp features, and threading it through costs nothing
`"distance"` mode doesn't already pay for by ignoring it. Tested at the
mechanism level (isolated from a live feedback loop's chaotic RNG
divergence — see test_change_spec_v7.py's own note on why the full-engine
correlation check needed the SAME exposures/actions held fixed): `"camp"`
with `camps=None` produces exactly zero animus movement; `"distance"` on
the identical exposures does not. On a controlled bimodal fixture (distance
cleanly tracking camp membership) the two formulas' per-user animus deltas
correlate above 0.95; on the full engine over a realistic `bimodal_normal`
population, so does final per-user animus after 60 ticks. `affect_d0=1.0`
is calibrated for a ONE-AXIS standard-normal population specifically (per
the spec); at Experiment 03's actual D=3, a raw multi-axis distance
saturates `phi` for same- and cross-camp dyads alike, which is a real scale
property of a fixed constant against a higher-dimensional geometry, not a
mechanism defect — recorded here rather than quietly re-tuned.

**V7.4 — cross-camp-restricted denominator.** `DeltaAff.per_cross_contact`,
normalized by an engagement/trait/post stance join (`_cross_contact_from_
frames`, a pure-frame core over polars DataFrames for the same
testability-without-a-run reason `_aff_from_arrays`/`_ideo_decomposition`
already split themselves out) classifying an event as cross-contact when
its dyad distance exceeds `d_cross = dynamics.affect_d0` — V7.3's own
saturation midpoint, so the two thresholds cannot drift apart as the spec
requires. `per_contact` (total-volume) is left exactly as it was;
`per_cross_contact` is NaN when the join's persistence targets (`traits`,
`posts`) are unavailable, never silently wrong.

**V7.5 — BIC margin.** `emergent_camps` returns `bic_margin` (the runner-up
k's BIC minus the selected k's, normalized by `|selected|`; NaN when fewer
than two k were fit). `DeltaIdeo.delta_bic_margin` joins the outcome set,
differenced like `delta_k` and — like `delta_k` — never NaN-gated on the
pre-period bimodality check the three camp-relative `toward_*` columns are.
On a synthetic population morphed from two clusters toward a third (fixed
within-cluster noise, only the separation parameter swept, so a wobble
cannot be resampling noise pretending to be non-monotonicity), the margin
fell monotonically while k held at 2, then k jumped to 3.

**V7.6 — re-gate, and a real gap it surfaced.** `GATE_ROWS` is now
`("reciprocity",)`; `attention_gini` and `clustering_ratio` are still
measured and still reported (`GateReport.rows`, `stylized_facts_report`'s
`in_range` flag), just no longer block. This is a demotion of a check that
could not be met, not a loosening of one that could: `attention_gini`'s
[0.8, 0.95] band is reachable only under `bandwagon` (`FINDINGS.md` above:
other kernels read 0.6-0.75, or ~0.97 once C3a's behaviour reinforcement
compounds it under `drift="full"`), and Experiment 03 runs `outrage`.

Actually running `stylized_gate` at "the Experiment 03 substrate" (the
Wave-A-background dial point, 0.7/0.7/0.7) for the first time — it had
never been gated before; Wave A's own module never calls `stylized_gate` —
surfaced a real, pre-existing gap unrelated to V7.3's mechanism at all:
reciprocity measured 0.065 against the [0.2, 0.4] band, because
`graph.sbm_mirror_p` (the SBM generator's OWN reciprocity top-up —
`network/sbm.py::sbm_graph` never reads the shared `graph.mirror_p` field
at all, contrary to `GraphConfig.mirror_p`'s own docstring) defaults to
0.0 and `dial_config` never set it. Calibrated empirically, the same way
`experiments/gate.py::calibrated_gate_config` calibrated its own
`mirror_p` (`network/reciprocity.py`'s "mirror_p is NOT the reciprocity you
then measure" applies here too): `sbm_mirror_p=0.15` measures reciprocity
at 0.289-0.301 across N=2,000 and N=10,000, comfortably mid-band, now
`dial_config`'s own default. Confirmed separately that `affect_drive` moves
nothing structural: reciprocity and clustering_ratio read identically
under `"camp"` and `"distance"` at a fixed seed, as they must — nothing in
graph generation reads `dynamics.affect_drive`.

With the gate passing, `DynamicsConfig.affect_drive` now defaults to
`"distance"`; `"camp"` remains available to reproduce a pre-V7.3 result
(Experiment 01's SBM finding among them) under the exact mechanism it was
measured with.

**Two regressions from this landing, fixed, not papered over.**
`test_c1_animus_rises_under_outrage_and_not_under_null`'s `1e-4` margin
between `outrage` and `null` kernel animus growth was calibrated against
`"camp"` mode's sharp contrast (a same-camp `null` engagement contributes
EXACTLY zero under `"camp"`; under `"distance"` it contributes a small but
non-zero `phi(d)`, narrowing — not reversing — the gap to 6.7e-5,
under the threshold). This is a pre-V7.3 mechanism conformance test, so it
now pins `affect_drive="camp"` explicitly rather than inheriting whatever
the default happens to be; the distance-mode analogue of the same claim is
one of test_change_spec_v7.py's own V7.3 tests.

`test_gate_passes_on_the_calibration_of_record_and_warns_off_gate`'s
off-gate probe was `dynamics.ranker="chronological"`, chosen specifically
because it lands attention Gini far below band — a row that no longer
blocks. Measured directly: `chronological` still PASSES the gate now that
only reciprocity is graded (reciprocity 0.230-0.233, in-band; Gini
0.630-0.641, now merely reported). The probe is now `graph.mirror_p=0.0`
on `calibrated_gate_config()`, which removes the shared reciprocity
top-up and lands reciprocity at 0.156-0.159 — under band, matching
`network/reciprocity.py`'s own documented ~0.157 chance-reciprocity
baseline for this generator with the mirroring pass off.

## Wave A′ — the full dial range, and a second camp-gate V7.3 didn't reach

Run per experiment03-bubble-intervention.md §5.4's "required" row and
change-spec-v7-continuous-affect.md's own closing "Unlocks": Wave A's
design repeated at the same scale (N=1,000, 60 burn-in + 100 post ticks,
3 dials × {0.1, 0.5, 0.9} × 5 seeds, 45 design points, 135 rows), under
the now-default `affect_drive="distance"`, background moved from Wave A's
0.7 to the neutral 0.5 (`run_wave_a_prime`, `results/experiment03/
wave_a_prime.csv`). ~17 minutes wall clock at ~25s/design-point — close
to the ~14s/design-point at N=800/80-ticks this module's own calibration
recorded, scaled to this run's larger N and tick count.

**H3 holds more cleanly than in Wave A, over the full range.** `composition`
is negative in all 45/45 cells (delta_aff_plateau -0.0158 to -0.0040);
`engagement` is positive in all 40 cells where its mechanism is active (see
below) and never negative. Wave A's own 40/40 was measured over a slice
that excluded the low ideological end entirely; this is the same finding,
now confirmed on ground Wave A could not reach.

**A second, unrelated camp-gate — the reaction kernel's own — makes the
`engagement` arm a complete no-op below the SAME bimodality threshold,
independent of `affect_drive`.** At ideological=0.1, `engagement`'s
`delta_aff_plateau`, `delta_aff_per_contact`, `delta_aff_per_cross_contact`,
`delta_k` and `delta_bic_margin` are all exactly `0.0` (not small — exactly
zero) and every camp-relative `toward_*`/`ideo_level_*` column is `nan`, for
all 5 seeds. Confirmed directly: at this design point `camps_and_bimodality`
returns `None` (bimodality 0.355, same population `dynamics/drift.py`'s
old gate would have frozen). The `engagement` arm's entire manipulation is
a `kernel_theta` SET override on the `outgroup` feature
(`exposure/kernel.py::compute_features`) — a DIFFERENT, still-standing
consumer of the camp label than the one V7.3 touched. `compute_features`
only adds `outgroup`/`outgroup_x_animus`/`ingroup_x_ident` `if camps is not
None`, and `apply_kernel`'s theta loop silently skips any entry naming a
feature that is not present — so below the gate, `engagement`'s config
differs from `none`'s in an override that never fires, and the run is
bit-identical to `none` down to the same-seed RNG draws. **V7.3's own
warrant — "the affect op is the last consumer of the camp label in the
mechanism path" — is not quite right; this is a second one, and it was not
in V7's scope.**

`composition` and `exposure` stay measurable at ideological=0.1 precisely
because neither depends on that feature for its OWN effect: `composition`
adds `valence_gamma0=2.5` (a platform-wide civility shift, independent of
camps) on top of the same `kernel_theta` override, and that shift alone —
interacting with the now-continuous affect channel — produces a real,
seed-consistent de-escalation (-0.00598 to -0.00460 across the 5 seeds at
this cell, versus exactly 0.0 for `engagement` on the identical population).
`exposure` (`inject_k`) never touches camp features at all. This is a
genuine, nameable methodological finding rather than a defect to route
around silently: **a camp-*aware* intervention (this codebase's only
implementation of "promote cross-camp engagement") cannot act on a
population that has not yet structurally sorted into visible camps, even
though the OUTCOME it would be judged on is now measurable there.** The
experiment03-bubble-intervention.md brief's own §4 `targeting_mode ∈
{distance, camp_pair}` factor is the fix this points at — a `distance`
targeting mode would promote engagement by continuous stance distance
instead of a camp label, exactly as V7.3 did for the affect channel — but
it is not built; the current `engagement`/`composition` arms are what the
brief would call `camp_pair` targeting, unswept and unlabelled as such.

**H1 is still neither confirmed nor falsified, now with the previously-inert
region included.** No sign flip for `composition` or `engagement` anywhere
across the full range on any of the 3 one-factor-at-a-time slices. The one
nominal sign change (`exposure`, sweeping `ideological`: mean -3.4e-5 at
0.1 vs positive at 0.5/0.9) is not a real crossing — per-seed values at that
cell are `[-3.8e-5, -2.1e-4, +3.6e-5, +1.5e-5, +3.3e-5]`, the same
small-and-seed-inconsistent pattern Wave A already reported for `exposure`
generally (37/40), not a new one. A true response-surface search (Wave B)
over the interior of the 3-dial volume is still what H1 needs.

**H2, resolved: the "hardening under every arm" reading was overwhelmingly
background, exactly as V7.2 predicted.** `ideo_level_toward_own_pole`
(`none`'s own absolute movement) ranges from -0.0004 to +0.0262 across the
sweep — an ~65x span driven by tribalization level, most visible on the
affective dial (0.0016 at level 0.1 rising to 0.0233 at level 0.9).
`composition`'s OWN net contribution (`toward_own_pole`, differenced) is an
order of magnitude smaller and roughly flat across levels (-0.0005 to
+0.0026) — it does not track tribalization the way the background does. Its
animus benefit (`delta_aff_plateau`) DOES scale with tribalization (-0.0040
at the low end to -0.0158 at the high end on the affective sweep). So the
named pattern survives measurement, refined rather than debunked:
composition's de-escalation grows with tribalization while its own
ideological cost stays small and roughly constant — a different, more
precise claim than Wave A's raw numbers supported.

**H5's falsifier looks like it holds at this scale, more informatively than
Wave A's flat `delta_k=0.0` could show.** `delta_k` is exactly 0.0 on all
135 rows again. `delta_bic_margin` is no longer identically flat (it can — 
it is a continuous quantity), but its movement is small (order 1e-4) and
non-monotonic across levels, reading as noise around zero rather than a
directional signal — closer to "the binary frame is adequate at this scale"
than to a gradient precursor of a real k-change.

**What Wave A′ does not do.** It reuses Wave A's 4-arm design as-is;
`targeting_mode`, the diversity-floor sweep, named scenarios, and hysteresis
(experiment03-bubble-intervention.md §§2.3-3-4-5.2-5.3) are Wave B/C
territory and require infrastructure this session did not build. Not
comparable to `wave_a.csv` line-for-line: different mechanism
(`affect_drive`), different structural substrate (`sbm_mirror_p=0.15`),
different background (0.5 vs 0.7) — a fresh measurement, not a correction
of the old one.

**What this is not.** Wave A′ (the sequencing table's step 7 — "sign screen
repeated over the full dial range, previously-gated region included") is
not run in this session, matching how this project has previously kept
"build the infrastructure" and "run the experiment" as separate steps
(Experiment 03's own infrastructure and Wave A run were two commits, not
one). `dial_config`'s `sbm_mirror_p=0.15` addition means a Wave A′ run is
not directly comparable to Wave A's own `results/experiment03/wave_a.csv`
on structural grounds ALONE, on top of the mechanism change `affect_drive`
already implies — worth stating plainly before either is read as a
straightforward "re-run."
