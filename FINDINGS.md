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
averaging, THEN differenced arm minus none the same way `delta_aff` is --
`_ideo_decomposition`'s `net = _movement(stance1_arm) -
_movement(stance1_none)`, present since the infrastructure commit, before
Wave A ever ran; every `toward_*` number below is the arm's net
contribution over whatever `none` itself does, never each arm's raw
movement on its own) are reported as a pair, never collapsed —
`delta_ideo`'s three camp-relative components are computed against a FIXED
axis and camp split taken from the shared pre-intervention state, so a
later tick's population is scored against a reference frame that does not
itself drift.
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

**Standing caveat for every seed-cell count in this document (§7's own
"cluster by seed"), stated once here rather than re-derived at each site.**
The 5 seeds at one design point share that point's population and graph —
only the per-tick RNG differs — so "40/40 seed-cells" is 8 populations
measured 5 times each, not 40 independent draws. A count like this is
evidence about consistency WITHIN the populations sampled, not a
40-observation estimate of a population-level rate; every "n/n" figure
below (Wave A, Wave A′, Wave B, Wave C alike) should be read that way.

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
quantitatively rather than by construction. Precisely stated, since
`base_config` sets `valence_mode="endogenous"` (V3) for both arms: they
share the SAME `kernel_theta` change, and differ only in `valence_gamma0`
— the intercept of the endogenous civility response, `P(civil) =
sigma(gamma0 + gamma_animus*animus_i + ...)`, read off the ENGAGING user's
own animus. Under that response, `engagement`'s positive sign is not "no
civility mechanism" versus composition's "civility mechanism" — both arms
run the identical animus -> civility -> sign -> animus loop the V3
bistability probe already characterizes elsewhere in this document;
`engagement` runs it at `gamma0=0.0` (P(civil)=0.5 at animus=0) with MORE
cross-camp contact feeding it, so the loop operates comparatively
unopposed, while `composition` raises `gamma0` to 2.5 (P(civil)=0.92 at
animus=0), damping it. `exposure`'s small, occasionally-ambiguous effect also matches prior
work: Experiment 01 measured `inject_k` moving the raw cross-cutting-
exposure aggregate by almost nothing while the injected items themselves
were far more cross-cutting — "drowned out by follower fanout at any
dosage a platform would ship."

**The per-contact normalization gap is fatal to one specific claim in this
section, not to the section generally.** `delta_aff.per_contact`
normalizes by TOTAL engagement volume, not by cross-camp contact
specifically (the literal §2.2 ask). That is harmless to the H3 contrast
just stated: `composition` and `engagement` share the identical
`kernel_theta` change, so cross-camp contact volume is approximately
fixed BETWEEN them, and the sign flip is not a dosage artifact. It is NOT
harmless to `engagement` vs. `none` on its own (the "backfire," positive
40/40 result two paragraphs up): `engagement` raises cross-camp contact
BY CONSTRUCTION relative to `none`, so an animus delta normalized by total
volume is partly measuring dosage rather than the outrage-per-contact
this arm is supposed to isolate — exactly the tautology §3 of the brief
warns against, undefended here. V7.4 (change-spec-v7-continuous-affect.md)
has since built the engagement/author-camp join this needs
(`delta_aff.per_cross_contact`, `_cross_contact_from_frames`), motivated
by this exact gap; the re-instrumented replication below applies it
directly to Wave A's own design points.

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
this data — smaller than it first looked, and real.** `toward_own_pole` is
positive in 113/120 non-degenerate seed-cells, for EVERY arm — including
`composition`, which simultaneously LOWERS animus. (`toward_mean` negative
and `toward_other_camp` positive track it near-exactly at this camp
structure — close to mirror images of the same signal in a near-linear
two-camp setup, recorded in the outcome-pair's own module docstring — so
`toward_own_pole` alone, not a three-way conjunction, is the count that
follows; the other two are not three independent confirmations of the
same finding.) The brief's own §2 table anticipates two rows for the
ideological axis ("positions converge" vs "positions unchanged"); what
Wave A actually shows is a third pattern the table does not name:
positions do not converge OR stay put, they measurably move toward
radicalization even under the arm that reduces hostility.

Is that the intervention's own doing, or the substrate radicalizing on its
own regardless of arm? `toward_own_pole` is ALREADY `arm minus none`
(`_ideo_decomposition`'s `net = _movement(stance1_arm) -
_movement(stance1_none)`, present since the pre-Wave-A infrastructure
commit, before Wave A ever ran) — not each arm's raw movement — so a
positive value already means the arm radicalizes MORE than the shared
background does, not merely that it fails to arrest a background trend.
But "already differenced" does not mean "large": a re-instrumented
replication of Wave A's own design (identical dials/levels/seeds/
background=0.7/mechanism `affect_drive="camp"`, on the current corrected
substrate -- `dial_config`'s `sbm_mirror_p=0.15` fixes a reciprocity-gate
failure Wave A's own original substrate had, found and fixed by V7.6
after Wave A ran, so this is not a byte-identical rerun --
`results/experiment03/wave_a_reinstrumented.csv`, 135 rows, ~23 min)
reads `none`'s own `toward_own_pole` drift (`ideo_level_toward_own_pole`,
V7.2) ALONGSIDE the net effect for the first time. §7's own reference is
per design point, not pooled (A5 of this project's own pre-submission
checklist): the ratio net/background computed AT EACH of the 40 points,
then summarized, not medians-of-each-pooled-then-divided (the latter gave
8.8%/10.0%/17.8% in an earlier pass of this section — close enough not to
change the qualitative picture, but the per-point version below is the one
this section now stands behind). Per-point ratio is strongly right-skewed
(a handful of points where `none`'s own drift is small enough to blow the
ratio up dominate the mean; median is the number to read): median
9.9% (`composition`), 9.4% (`engagement`), 22.9% (`exposure`) of `none`'s
own drift. Against the brief's own §7 SESOI (10%), BOTH `composition` and
`engagement`'s own ideological contributions now fall (narrowly) short —
only `exposure` clears comfortably. Sign-consistency itself replicates
closely (111/120 net-positive across the three arms here, against 113/120
originally — the small gap is consistent with the substrate difference,
not a discrepancy needing its own explanation). **Revised claim: the
positive sign on `toward_own_pole` is a genuine net effect, not
background drift misread as one, but "hardening," as a named phenomenon
worth its own title, overstates a contribution this close to the brief's
own noise floor for `composition` AND `engagement` alike — only
`exposure`'s own ideological contribution is unambiguously large enough
to deserve one.**

**SESOI, applied to `delta_aff_plateau` itself (§7's decision rule, not
previously checked against any number in this section).** Sign
consistency alone is cheap when an effect is tiny but systematic (the
standing seed caveat above applies to every count in this paragraph too).
Per-point ratio (`|delta_aff_plateau| / (0.10 * |delta_aff_level_none|)`,
`delta_aff_level_none` new this pass, mirroring `ideo_level_*`'s
convention), each of the 40 non-degenerate points scored against its OWN
`none` arm at the SAME point, from the re-instrumented replication:

| arm | median ratio vs. SESOI | points clearing | verdict |
|---|---|---|---|
| composition | 12.8x | 40/40 | CLEARS at every point |
| engagement | 4.7x | 40/40 | CLEARS at every point |
| exposure | 3.4x | 34/40 | clears at most points, not all |

`composition` and `engagement` clear SESOI at every single design point,
not just on aggregate — the strongest form this check can take.
`exposure` clears comfortably in the median but NOT uniformly: 6/40 points
fall short, unlike the other two arms — a real, more precise finding the
earlier pooled-ratio version of this table (10.5x/4.6x/2.5x on medians-of-
medians, all reported as uniformly "CLEARS") did not have the resolution
to see. **H3 "holds cleanly" survives contact with its own decision rule,
uniformly for `composition` and `engagement` and at most points for
`exposure`** — unlike the `toward_own_pole` contribution above,
`delta_aff_plateau` is not a small effect relative to what the brief
treats as meaningful; it is the primary, well-powered outcome this design
was built to measure, and the ideological side-effect above is the
secondary one that is not.

**The dosage-confound check above does NOT resolve the confound — checked
directly (A4/B3 of this project's own pre-submission checklist), and this
reverses the previous paragraph's conclusion.** `delta_aff_per_cross_
contact`'s sign matching `delta_aff_per_contact` and raw `delta_aff_
plateau` (40/40 `engagement`, 0/40 `composition`, 37/40 `exposure` — same
counts either way) is NOT independent confirmation, because the two
denominators are not independent: classifying an event cross-contact when
its dyad distance exceeds `d_cross = affect_d0 = 1.0` selects **80-85% of
ALL contact, for every arm**, across every design point in this design
except one — at `ideological=0.1` specifically it drops to 57%, still a
majority (`results/experiment03/wave_a_reinstrumented_cross_contact_
fraction.csv`, all 45×4 arm/none combinations audited directly). A
denominator selecting the large majority of everything reproduces the
unrestricted denominator's result by construction (A4's own named failure
mode) — this is not a robustness check that happened to agree, it is close
to the same measurement asked twice. This is not a new surprise: V7.3's
own section already recorded that `affect_d0=1.0`, calibrated for a
ONE-AXIS population, saturates `phi` for same- and cross-camp dyads alike
at Experiment 03's actual D=3; this is that same miscalibration's
consequence in `d_cross` (V7.4 deliberately ties the two so they cannot
drift apart, which also means neither can be wrong without the other
following). **Revised claim: whether `engagement`'s backfire is a dosage
artifact remains OPEN at this substrate — untested, not negatively tested
(the C2 distinction, applied here) — because the one check built for it
does not discriminate cross-camp from same-camp contact well enough to be
informative.** A real test needs `d_cross` recalibrated for D=3 (e.g. to
the level that actually splits contact close to the true camp-membership
boundary, not a nominal midpoint carried over from D=1) or a same-camp-vs-
cross-camp classification that does not route through a single scalar
distance threshold at all — neither is built; out of scope for this pass,
recorded here rather than left for the normalization-invariance claim to
imply a resolution that was not actually reached.

**B1 of this project's own pre-submission checklist, applied properly: not
"already flagged," actually measured — and corrected once more after this
measurement's own first pass got the UNITS wrong.** V7.3's own section
states `phi` "saturates for same- and cross-camp dyads alike" at D=3 —
true of the absolute level, imprecise about the mechanism: measured
directly (join every `distance`-mode engagement event against the engaging
user's and post author's stable pre-period camp label, not just distance),
`phi` DOES discriminate true camp membership, and the gap widens with
tribalization rather than staying fixed. **Corrected numbers** (this
section's own first pass reported `phi` 0.666/0.832 and mean distances
2.29/5.27 at background=0.7, computed from RAW Euclidean distance; the
actual mechanism reads `stance_distance` RMS-normalized —
`dynamics.agreement_metric="rms"`, the default, divides by `sqrt(D)` —
so those numbers were too large by a factor of `sqrt(3)≈1.73`, caught
when work-order-01's own `_camp_boundary_d_cross` produced a value that
silently under-classified almost everything until the same unit mismatch
was found there too): mean `phi` 0.543 (true same-camp) vs. 0.742 (true
cross-camp) at background=0.7 (0.531 vs. 0.684 at 0.5; 0.556 vs. 0.786 at
0.9); mean RMS-normalized distance 1.32 (same-camp) vs. 3.05 (cross-camp)
at 0.7. The fraction near-saturated (`phi` > 0.8) now reads the opposite
of dramatic at the low end and still tells a real story at the high end:
0.09% of true same-camp contact vs. 12.0% of true cross-camp contact at
background=0.7, rising to 41.0% cross-camp (same-camp stays under 0.1%
throughout) at 0.9 — `phi` is essentially NEVER near-saturated for
same-camp contact at any tribalization level tested, contradicting
"saturates... alike" more directly than the first pass's own wrong-scale
numbers did. What IS true, and is the actual mechanism behind `d_cross`'s
80-85% selection-fraction figure above (which used the correct RMS scale
throughout, via the pre-existing `_cross_contact_from_frames` convention —
only this section's OWN new distance sampling had the bug): mean
same-camp distance (1.32 at background=0.7) already exceeds
`affect_d0=1.0`, so `d_cross=affect_d0` sits inside or past the
lower shoulder of the curve for the same-camp class too, not just deep in
cross-camp's saturated region — a threshold set there cannot cleanly
separate the two even though the underlying continuous signal is
discriminating. **Revised finding, more precise than either "saturates
alike" or this section's own first correction: the continuous mechanism
carries real information about camp membership at this D=3 substrate,
concentrated in the same distance RANGE `affect_d0=1.0` already sits in
(same-camp is not deeply saturated the way the first pass reported) — the
threshold inherited from D=1 does not discriminate well not because it is
too small for the population's scale, but because same- and cross-camp
distance distributions are not separated enough on the SAME scale `d0`
already lives at.** `d_cross`, calibrated by `_camp_boundary_d_cross` to
the realized midpoint between the two classes (≈1.98 at background=0.5,
≈2.40 at 0.7, ≈2.84 at 0.9 — computed from actual engagement pairs; a
uniform-population-pairs sample gives a close but not identical ≈2.2 at
0.7, the expected difference between sampling frames) sits closer to
`affect_d0=1.0` than this section's first pass believed, not the "nearer
3.5-4.0" it wrongly suggested.

**H5 (fragmentation is a distinct outcome) has no support in this pass.**
`delta_k` is exactly 0.0 on all 135 rows — `emergent_camps`' BIC-selected k
never moved, for any arm, at any design point. Could be a true negative
(the binary frame is adequate at this scale) or could be that 160 ticks
and N=1000 is simply too short/small for a k-means-BIC estimator to
register a shift — this pass cannot distinguish the two, and does not
claim to. What k itself is (not just its delta) at a few representative
points, checked directly against the re-instrumented replication's own
cached population states: pinned at k=1 at ideological=0.1 (bic_margin
~0.06, pre- and post-intervention alike — no camps to begin with, exactly
the gated region above), stable at k=2 at ideological=0.5 and 0.9
(bic_margin ~0.03-0.04, pre- and post- alike). So "no support" at
ideological=0.1 is "pinned at k=1," uninformative about fragmentation by
construction; at 0.5 and 0.9 it is "stable at k=2," which IS informative
— genuinely ruling out a shift toward more camps in the region where
camps exist, not silence from an estimator with nothing to say.

**What this is not, stated plainly (see also the module's own
docstring).** No hysteresis phase (§5.2) — nothing here says whether the
composition arm's de-escalation is reversible. No response-surface fit
over the full 3-dial volume, only three one-at-a-time slices through it.
No calibration against a corpus (§8) and no viewpoint-diversity floor
(§2.3) — both explicitly deferred as normative/scope choices in the brief
itself, unresolved here too. Waves B-D (the LHS response surface,
hysteresis, and held-out confirmation) are not run. The open decisions
§10 of the brief lists — the viewpoint-diversity floor's number,
corpus-or-swept-anchored-group, and the intervention target under
emergent k — are exactly as open as before this session; none of them
blocked Wave A, but all of them block Wave B.

**The bimodality gate is a real design constraint on Wave B, not only on
Wave A, and this pass's fix (background=0.7, clear of the gate) does not
generalize to it.** A naive LHS over the full 3-dial volume would put
points below the ideological gate where the affect channel is
IDENTICALLY zero for every arm (not low-signal — structurally constant),
which would fit a response surface across a step function rather than a
trend, and dominate its leverage. Resolved differently than either
obvious option (truncate the swept range, or fit only above the gate as
an explicit boundary): change-spec-v7-continuous-affect.md's
`affect_drive="distance"` mechanism replaces the discontinuity itself with
a continuous saturating function of stance distance, so the affect
channel has something to report on both sides of the old gate — this is
what Wave A′ and Wave B are built on (below). That fix is NOT complete for
every consumer of the camp label, though: Wave A′ separately found that
the `engagement`/`composition` arms' OWN `kernel_theta` override reads a
DIFFERENT, still-conditional feature (`outgroup`), so `engagement` under
`targeting_mode="camp_pair"` remains an EXACT structural zero below the
same gate — confirmed again in Wave B's own `low_tribalization` scenario.
Wave B's response-surface fit (below) sidesteps this by being built for
`composition` only, whose own lever (`valence_gamma0`) is not camp-gated
and so never hits an exact zero — not because the discontinuity was
modeled as a boundary, but because no regression was attempted over the
one arm/mode combination that still has one.

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
`posts`) are unavailable, never silently wrong. Tying `d_cross` to
`affect_d0` so the two cannot drift apart also means neither can be wrong
without the other following: the Wave A section below finds directly that
`affect_d0`'s own D=1-calibrated value (already flagged in V7.3, next)
makes `d_cross` classify 80-85% of ALL contact as "cross-camp" at this
D=3 substrate, not a restrictive subset — `per_cross_contact` does not yet
deliver the discriminating check §2.2 wants it for.

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

**B4 of this project's own pre-submission checklist, re-run rather than
assumed still complete: `grep -rn "camps is not None\|camps is None"
discourse_lab/` finds a THIRD site, not documented above —
`metrics/dmp.py`'s thread-friction computation.** Checked whether it
matters here: `dmp_table`/`dmp_thread_count` are never imported by
`experiment03_bubble_intervention.py` or its notebook — dead code with
respect to every claim in this document, not a third gate any Experiment
03 result passes through. The two consumers enumerated above remain the
complete set THIS experiment's results depend on; `dmp.py`'s own camp-gated
metric would need this same enumeration repeated if a future analysis
ever calls it.

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

**Against the brief's own §7 SESOI (10% of `none`'s own drift, per-point
ratio — see the same check applied to the original Wave A design, above):
all three arms clear it comfortably here.** Median ratio 22.8%
(`composition`, 42/45 net-positive), 24.1% (`engagement`, measurable
cells only, 36/40 net-positive), 22.7% (`exposure`, 42/45 net-positive) —
roughly DOUBLE the ~10% SESOI line, and roughly double what the original
Wave A design (camp-gated mechanism, background=0.7) found for
`composition`/`engagement` specifically (9.9%/9.4%, narrowly BELOW the
line). Same brief, same rule, two different verdicts — because the
mechanism and background differ, not because either measurement is wrong:
under the continuous `affect_drive="distance"` mechanism at the neutral
0.5 background, the ideological side-effect this section names is larger
relative to `none`'s own drift than it is under the old camp-gated
mechanism at 0.7. The finding this section's own bold claim rests on is
real by the brief's own decision rule at THIS mechanism/background; stated
without that qualifier, it would overclaim what generalizes.

**H5's `delta_k` component is untestable here, not falsified.** `delta_k`
is exactly 0.0 on all 135 rows again — a STRUCTURAL consequence of this
codebase having no population generator that produces more than two
ideological modes, true regardless of what any intervention does, not
evidence that fragmentation fails to respond to one. "Not supported" would
overclaim a look that never happened; there is no signal to read either
way. `delta_bic_margin` is the part that actually can speak, since it is a
continuous quantity NOT pinned to zero by the same limitation: it moves by
only order 1e-4 and non-monotonically across levels, closer to noise around
zero than to a directional signal — a weak lean toward "no gradient
precursor detected" on 135 rows at 5 seeds each, not a resolved result.

**What Wave A′ does not do.** It reuses Wave A's 4-arm design as-is;
`targeting_mode`, the diversity-floor sweep, named scenarios, and hysteresis
(experiment03-bubble-intervention.md §§2.3-3-4-5.2-5.3) are Wave B/C
territory and required infrastructure this session had not yet built (built
and run below). Not comparable to `wave_a.csv` line-for-line: different
mechanism (`affect_drive`), different structural substrate
(`sbm_mirror_p=0.15`), different background (0.5 vs 0.7) — a fresh
measurement, not a correction of the old one.

## Wave B/C — targeting_mode, the diversity floor, named scenarios, hysteresis

Built to answer Wave A′'s own closing finding (the `engagement`/
`composition` arms' `kernel_theta` reads `CONDITIONAL_FEATURES` and is a
complete no-op below the bimodality gate, independent of `affect_drive`)
and to run experiment03-bubble-intervention.md's §§2.3/3/5.2/5.3
(the diversity floor, named scenarios + LHS, hysteresis) for the first
time. Infrastructure (`targeting_mode`/`arms_for`, `dispersion`/
`diversity_ratio`/`diversity_floor_break_even`, `SCENARIOS`/
`lhs_design_points`, `run_hysteresis`/`run_wave_c`) and its conformance
tests (tests/test_experiment03.py) landed in one commit; this section is
the first RUN of it, against the full 364-test suite passing throughout.

**Wave B design, reduced the same way Wave A was reduced from a true
Morris design (module docstring): 3 named scenarios
(`consolidated_two_camp`, `cross_cut`, `low_tribalization`) × 6 Latin
Hypercube points per scenario (radius 0.25 around its dial preset) × both
`targeting_mode`s × 2 seeds — 72 design points, 216 rows
(`results/experiment03/wave_b.csv`), 1435s (~24 min) wall clock.** Per
design point cost ~26s for `targeting_mode="camp_pair"` and ~14s for
`"distance"` at the SAME (scenario, LHS point, seed) — not a
`targeting_mode` performance difference but `run_wave_b`'s own loop order:
the `none` arm never depends on `targeting_mode`, so the second mode's
`none` run at a given point is a `cached_run` hit. Sanity check before
anything else: the `exposure` arm is bit-identical across `targeting_mode`
in all 72 of its rows (`arms_for(...)["exposure"]` never references
`kernel_theta`, so its whole trajectory — same seed — is unaffected by
which theta table `targeting_mode` would otherwise select) — the factor is
wired where intended and nowhere else.

**H3b: `targeting_mode` moves `engagement` uniformly and `composition` by
regime, in OPPOSITE directions from each other.** Paired within every
(scenario, LHS point, seed) cell, `camp_pair`'s `delta_aff_plateau` is
LOWER than `distance`'s for `engagement` in **24/24 MEASURABLE cells**
(`consolidated_two_camp` and `cross_cut`, mean gap -0.0178 and -0.0091) —
counted separately from `low_tribalization`'s 12 cells, where `camp_pair`
is gated off (exactly 0.0 in all 12, per Wave A′'s own kernel_theta
finding above): those are not 12 more confirmations of the same
comparison, they are 12 cells where one side is a structural zero, listed
here rather than folded into "36/36" as if all 36 were equally informative
(A2 of this project's own pre-submission checklist, caught on a third
pass — this claim had stood since Wave B's own original write-up).
`distance` targeting always produces MORE hostility increase than
`camp_pair` on the cells where the comparison is real, not only where
`camp_pair` happens to be gated off. For `composition` the interaction
flips sign by regime: in `consolidated_two_camp` (strongly, structurally
tribalized) `camp_pair` reduces hostility MORE than `distance` (mean gap
-0.0081, 8/12 cells favor `camp_pair`); in `low_tribalization` `distance`
reduces it slightly more, with total consistency (12/12 cells, mean gap
+0.0010); `cross_cut` is weaker and less consistent (8/12 cells favor
`distance`, mean gap +0.0005). Camp-based and distance-based targeting are
not interchangeable implementations of "the same" manipulation — which one
does more, in which direction, depends on how structurally sorted the
population already is.

**H1: it DOES fire, once, and this was true of the ORIGINAL Wave B data
too — missed until work-order-01's own before/after check surfaced it.**
`composition` is positive at 2/72 rows: both seeds of
`consolidated_two_camp`, LHS index 1, `targeting_mode="distance"` — the
single most-tribalized corner of that scenario's own LHS neighborhood
(affective=0.946, ideological=0.885, structural=0.732), and NOT a
recalibration artifact (the pre-recalibration `wave_b.csv` has the
identical 70-negative/2-positive split at the identical point,
0.0099/0.0126 there vs. 0.0074/0.0085 post-recalibration — same sign,
smaller magnitude). `engagement` is non-negative everywhere (zero only in
the now-well-understood gated cells) across all 216 rows, extending Wave
A′'s 45/45. `exposure` stays small and sign-varies by scenario (positive
in `consolidated_two_camp`, mixed elsewhere). **Revised: H1's falsifier
has fired, narrowly — a real, seed-consistent sign crossing exists WITHIN
`consolidated_two_camp`'s own neighborhood, at its most extreme corner,
under `distance` targeting specifically (the paired `camp_pair` value at
the identical point stays negative, -0.0102) — not "not yet falsified,"
a genuine but narrow exception the two prior write-ups of this exact
dataset (the original Wave B commit, and this document's own Verdict
table) both missed.** This is a single design point out of 18 unique
`distance`-mode ones sampled, and a single scenario's own extreme corner,
not a broad crossing — the honest reading is "the sign is not universal,"
not "the sign is unstable."

**SESOI check on the fire itself (work order 02, task 3) — it clears, so
the "fires" language stands rather than needing to weaken to "flips below
the smallest effect of interest."** Same rule as elsewhere in this
section, applied at this one point (recalibrated data — `delta_aff_level_
none` does not exist in the original `wave_b.csv`, so only the current
numbers get this treatment):

| seed | `delta_aff_plateau` | `none`'s own drift | 10% SESOI line | ratio |
|---|---|---|---|---|
| 0 | 0.007368 | 0.019185 | 0.001919 | 3.84x |
| 1 | 0.008467 | 0.017549 | 0.001755 | 4.83x |

Both seeds clear SESOI comfortably (3.8x/4.8x), smaller than
`composition`'s typical double-digit multiple elsewhere in this table but
nowhere near the line — this is a real effect by the brief's own
decision rule, not one that only survives as a sign. **What DOES limit
this claim is seed count, stated rather than left implicit: this point
has 2 seeds, against the brief's own 10** — the reason its weight rests on
the cross-calibration reproduction two paragraphs up (same sign, same
point, both mechanisms) rather than on these two seeds alone. Two
consistent seeds under one calibration is a coincidence candidate; two
consistent seeds reproducing again under a materially different `d0`/
`d_cross` calibration is not the same claim.

**The diversity floor rarely gets tested at this scale, because dispersion
barely moves — checked with a sensitivity test this time, not just read
off a flat statistic (C5 of this project's own pre-submission checklist).**
90/216 rows reduced hostility at all (`composition` 70/72, `exposure`
20/72, `engagement` 0/72 — it never once reduces hostility in this run).
Among the 90, `diversity_floor_break_even` — SS2.3's break-even ratio `f`,
literally `diversity_ratio_arm` when the arm reduces hostility (§4's own
build note) — ranges 0.997 to 1.024. **Stated plainly, since it is easy to
misread from the number alone: this is a PRE-vs-POST ratio within the
arm's OWN trajectory, not an arm-minus-`none` difference** — unlike
`toward_own_pole`, it does not automatically net out shared background
movement.

Sensitivity check: does the same statistic move when something in this
data demonstrably does? `none`'s own version of it (`diversity_ratio_
none`, the null's pre/post dispersion ratio over the identical window)
is ALSO flat — 0.997 to 1.023 across all 216 rows — while `ideo_level_
toward_own_pole` (animus/ideology's OWN background drift, same rows) spans
a ~65x range. Dispersion is genuinely a far more stable summary statistic
at this population size and horizon than animus or ideology are; the flat
`diversity_ratio_none` is not proof the check is broken, it is evidence
there is little background movement HERE to net out in the first place —
a materially different situation from `toward_own_pole`'s, where a flat
number WOULD have been suspicious. Netting it out anyway (`diversity_
ratio_arm - diversity_ratio_none`, the arm's actual incremental
contribution over the SAME 90 hostility-reducing rows `diversity_floor_
break_even` is itself defined over — `engagement`'s own 0 qualifying rows
excluded, not folded in) surfaces a small effect invisible in the raw
ratio, but only for one arm: mean +0.0028 (`composition`, n=70,
consistently positive), -0.00006 (`exposure`, n=20, indistinguishable from
zero). `engagement` has no net-effect number here at all — it is undefined
under this statistic's own domain, the same reason it contributes nothing
to the 0.997-1.024 band above. **Revised verdict: SS2.3's central
question — is civility worth a diversity cost — still does not have a
real dilemma to adjudicate here, but for a checked reason now, and a more
specific one than "every arm barely moves it": `composition` (the only
arm both reliably reducing hostility AND newly checked here) shows a
small, real, consistently-signed net INCREASE in dispersion (a few tenths
of a percent) once background is subtracted out, `exposure`'s is
genuinely negligible, and `engagement` was never in scope for this
particular tradeoff to begin with.** A longer horizon or a stronger
intervention could change the magnitude; this run tests neither.

**Ideological movement stays dissociated from the affective outcome,
consistent with Wave A′'s H2 finding.** Where camp is defined
(`consolidated_two_camp`, `cross_cut`; `low_tribalization`'s 72 rows are
correctly NaN-gated, its pre-period being unimodal), `toward_other_camp` is
negative for every arm in both scenarios — the population moves AWAY from
the other camp's pre-period centroid regardless of arm, `composition`
included, even in the same cells where `composition` is reducing animus.
`delta_k` is exactly 0.0 on all 216 rows — structurally, not empirically:
`cross_cut`'s own documented no-k>2-generator limitation guarantees this
regardless of what the interventions do, so it is untestable here rather
than a finding of no effect. `delta_bic_margin`, which is NOT pinned by
that limitation, is small and non-directional (mean -0.00017 to +0.00005 by
arm) — the actual (weak) signal, matching Wave A′.

**The response surface itself — a revised target, not abandoned work.**
experiment03-bubble-intervention.md's §5.3 names a response-surface fit
over the LHS points as the deliverable, and it was not built: 6 points x 2
seeds per scenario cannot support fitting one, let alone the ~30-point,
10-seed design the brief itself specifies. H1's falsifier DOES fire once
in this data (above — a narrow exception found on a later pass, not
absent), but a single design point at one scenario's own extreme corner is
not a crossing a PLANE could usefully locate — a linear surface fit
against 18 points has no way to represent an exception at one corner
without also distorting its fit everywhere else, so the surface's job
changes for a related but distinct reason than "there is no crossing":
there is one, but it is not surface-shaped. What is left, and
is a smaller, tractable ask on exactly the same LHS points, is mapping HOW
`composition`'s hostility reduction scales across the 3 dials, rather than
reporting Wave A′'s two `affective`-sweep endpoints (-0.0040 at level 0.1,
-0.0158 at level 0.9, a ~4x range) as if they were the whole shape.

A plain linear surface (`delta_aff_plateau ~ affective + ideological +
structural`, OLS, no interaction terms — 18 LHS points is not enough
degrees of freedom to support them), fit separately per `targeting_mode`
since H3b already found the two are not interchangeable:

  `camp_pair`: intercept -0.0001, affective -0.0170, ideological -0.0102,
  structural +0.0084, R^2 = 0.908.
  `distance`:  intercept -0.0054, affective +0.0084, ideological -0.0199,
  structural +0.0122, R^2 = 0.415.

**R^2 = 0.91 is not evidence on its own, and this project's own
pre-submission checklist (D1, blocking) says why: `camp_pair`'s 18 points
sit in exactly 3 neighborhoods (one per scenario, LHS radius 0.25 around
each center), so this is a 4-parameter fit against 3 effective design
points at the between-cluster level — under-determined by the checklist's
own criterion, not merely coarse.** Checked directly, not just flagged:
a model using ONLY which scenario a point belongs to — no dial values at
all — already reaches R^2 = 0.844, 93% of the full fit's 0.908. Refit with
scenario fixed effects (so the 3 dial slopes come from WITHIN-cluster
leverage only, the between-cluster mean absorbed separately): affective
-0.0130, ideological -0.0010, structural +0.0052 — ideological's slope
collapses to near-zero (the plain fit's -0.0102 was mostly the 3 scenarios'
own mean differences, not a within-scenario ideological trend), while
affective survives at roughly its original size AND stays close to the
~-0.0148/unit slope Wave A′'s own one-factor sweep measured independently
(-0.0118 over 0.8 units) — the one part of this surface with a real
out-of-sample cross-check behind it. Leave-one-scenario-out prediction
(fit on 2 scenarios, predict the third) is the sharper test: R^2 = 0.072 —
barely above predicting the global mean, and far below what R^2 = 0.908
implies about this fit's ability to generalize. **Revised verdict (work order 02, task 5): the coefficient table above is
NOT INTERPRETABLE, not merely caveated.** This project's own D1 (pre-
submission checklist, blocking) is why the distinction matters: a fit
must contain the mechanism it predicts about, and a 4-parameter surface
fit to 3 effective design points does not, regardless of its in-sample
R². "Not a genuinely good linear fit" understates it — the coefficients
(`camp_pair`'s affective -0.0170, ideological -0.0102, structural
+0.0084; `distance`'s +0.0084/-0.0199/+0.0122) are not a usable
description of what the three dials do and should not be read as one,
independent of the ONE narrower fact that survives: `affective`'s slope
under `camp_pair` has independent corroboration from Wave A′'s own
one-factor sweep, which is a claim about that single dial, not about the
table as a whole. `distance`'s surface (R^2 = 0.42) was already reported
as a plain, honest negative result — the same leave-one-scenario-out
check makes it more decisively so (R^2 = -8.96, worse than the mean),
consistent with H3b's own finding that `distance` targeting's effect on
`composition` changes SIGN between regimes (`consolidated_two_camp` vs
`low_tribalization`), which no hyperplane fit to 3 points can represent
regardless of R^2.

**Independently replicated, not just argued once: refitting the identical
18-point design at work order 01's recalibrated `affect_d0`/`d_cross` made
BOTH leave-one-scenario-out numbers WORSE, not better** —
`camp_pair` 0.072 -> -0.455, `distance` -8.99 -> -9.26 (Decision 1d).
Negative LOO R^2 means the fit predicts worse than the held-out scenario's
own mean; recalibration touches the affect channel and the analysis-only
`d_cross` classifier, nothing about how `run_wave_b`'s 18 points are laid
out, so an under-determined design behaving unpredictably across two
materially different refits of the SAME data is exactly what D1 predicts,
observed twice rather than asserted once.

Caveat this fit shares with everything else built on Wave B: 18 points
sitting in 3 tight, radius-0.25 neighborhoods around 3 centers is not a
space-filling design over the cube, so this is a coarse global trend
across the sampled region, not a validated interpolant between it —
adding interaction or quadratic terms would need more points than this
run has, not a modeling choice deferred out of laziness.

**Wave C: hysteresis on the 3 largest Wave B effects — all in the
direction where the intervention makes things worse.**
`select_hysteresis_points` (seed-averaged `|delta_aff_plateau|`, top 3)
picked `engagement`/`targeting_mode="distance"` at all 3 points — 2 in
`consolidated_two_camp` (LHS 1 and 5), 1 in `cross_cut` (LHS 0) — because
`engagement`'s hostility INCREASE under `distance` targeting is larger in
magnitude than `composition`'s hostility decrease anywhere in this run.
Run at N=1,000, 60 burn-in + 60 pre-withdrawal + 60 post-withdrawal ticks,
5 seeds per point (15 hysteresis runs, `results/experiment03/wave_c.csv`,
195s). **Recovery is partial and strikingly consistent across all 3
points: `recovery_fraction` 0.575-0.658 per run** (mean 0.628 ±0.027 at
`consolidated_two_camp` LHS1, 0.635 ±0.025 at LHS5, 0.612 ±0.023 at
`cross_cut` LHS0) **— roughly 61-64% of the peak animus gap closes in the
60 ticks after withdrawal, leaving 36-39% persistent.** Neither H4 extreme
holds cleanly: not fully sticky (0%), not fully reversed (100%), at a 1:1
withdrawal-to-intervention tick ratio. The tightness of the cluster across
two scenarios and 3 dial points, despite peak gaps varying 2.5x (0.0151 to
0.0371), is not a pattern to hypothesize about — this codebase already
specifies the mechanism it comes from, and checking it against the actual
numbers is cheap.

**Checked, not just hypothesized: the recovery IS the OU mean-reversion
rate `dynamics.affect_ou_k`, arithmetic away.** `apply_drift`
(`dynamics/drift.py`) updates animus each tick as `X_{t+1} = X_t + gain -
k*(X_t - Bs_t) + noise`, `Bs_{t+1} = Bs_t + k_b*(X_t - Bs_t)`, with
`k = affect_ou_k = 0.02` and `k_b = k/10` (`block_rates`). After
`withdrawal_tick` both arm and none runs share the identical schedule, so
their exogenous `gain` terms are the same process and cancel in the
DIFFERENCE — the gap `(Delta, DeltaBs) = (X_arm - X_none, Bs_arm - Bs_none)`
then evolves by the pure LINEAR, homogeneous 2x2 recursion `v_{t+1} = M
v_t`, `M = [[1-k, k], [k_b, 1-k_b]]`, with no free parameters beyond the
two the config already names.

The single-timescale shortcut (treat `Bs` as fixed, `Delta_t = (1-k)^t
Delta_0`) predicts recovery `1 - (1-k)^60 = 0.7024` at the 60-tick
withdrawal window — noticeably ABOVE all 3 observed values (0.605-0.635),
which is itself informative: FINDINGS.md already documents (V3 section)
that `Bs` drifting toward `X_stored` at `k_b` is what removes the
self-reinforcement loop's stable fixed point, i.e. `Bs` is never really
fixed on the timescales this codebase's own dynamics operate at. Fitting
the FULL 2-state recursion instead — `Delta_120` fixed at the measured
peak gap, `DeltaBs_120` (the `Bs` gap already accumulated by the time of
withdrawal — not directly observable, since `Bs` is engine-internal state,
never persisted) as the ONE free parameter, least-squares against each
point's own full 60-tick post-withdrawal trajectory (mean of the same 5
seeds `run_wave_c` used) — recovers a small, physically sensible
`DeltaBs_120` at all 3 points (5-9% of that point's own peak gap, the right
order of magnitude given `k_b` is 10x slower than `k`) and an R^2 above
0.999 at every point. The 2-state model's own predicted recovery —
0.6296, 0.6321, 0.6047 — lands within 0.001-0.008 of the actually observed
0.6217, 0.6271, 0.6059. That gap is within what 5-seed averaging noise
alone would produce. So: not a fixed-timescale coincidence needing
further explanation, and not quite the naive single-exponential either —
recovery here is the two known OU constants this experiment never
touched, doing exactly what `drift.py` says they do.

**The other side of H4's own comparison: `composition`'s benefit recovers at
the SAME rate as `engagement`'s harm — no asymmetry.**
`select_hysteresis_points` now takes an `arm` filter (previously it always
surfaced whichever arm had the single largest-magnitude effect, which was
`engagement` in every one of Wave B's 216 rows) precisely so H4's
"stickier hostile than civil" claim could be checked on its own intended
BENEFIT side, not inferred from having only measured the harm side. Top-3
`composition` points by |delta_aff_plateau| (`select_hysteresis_points
(wave_b, n_points=3, arm="composition")`) are all in `cross_cut` (LHS 0
camp_pair, LHS 4 camp_pair, LHS 4 distance) — the scenario where
`composition`'s own effect happens to be largest, unlike `engagement`'s
top-3 which spanned two scenarios. Same run shape as before (N=1,000,
60/60/60 burn-in/pre/post-withdrawal ticks, 5 seeds,
`results/experiment03/wave_c_composition.csv`, 214s):

  `composition` (benefit): recovery_fraction mean 0.6078, SD 0.0270, range
  0.559-0.664 (n=15).
  `engagement` (harm, `wave_c.csv`): recovery_fraction mean 0.6250, SD
  0.0253, range 0.575-0.658 (n=15).

The two distributions overlap almost completely — a ~0.017 difference in
means against a ~0.026 SD on each side is noise, not an effect. **This is
expected, not merely unremarked, given the mechanism check above**: the
`(Delta, DeltaBs)` recursion `drift.py`'s OU reversion produces is LINEAR
and sign-symmetric in the gap — `k`/`k_b` do not know or care whether
`Delta` is positive (harm) or negative (benefit), so a model where recovery
comes from that mechanism predicts equal recovery rates in both directions,
which is exactly what both runs show. **At this comparison's own scale
(60-tick pre-withdrawal), H4's asymmetry claim ("the hostile regime is
stickier than the civil one") is not supported** — partial persistence
(the non-asymmetry-specific half of H4) holds in both directions, at a
rate this project can now name from the OU constants alone rather than
measure arm by arm.

**Superseded, not merely re-checked: work order 02's gap-ladder test
(below, and in full under "Work order 02, task 1") found the opposite at
the SAME points, run jointly.** Two defects in the comparison just
above — harm and benefit measured at DIFFERENT design points, and
statistics pooled across 3 non-independent points — meant it could not
actually distinguish "no asymmetry" from "an asymmetry too small to see
through those two flaws." Fixing both (a common 3-point design, per-point
statistics) and adding a 60/300/1000-tick ladder finds a REAL, if narrow,
asymmetry: relative asymmetry grows with exposure duration at 2 of 3
points, clearing a pre-registered 3-sigma bar at both. Work order 01's
own larger-gap check (decision 2, immediately below) inherited both of
this comparison's defects and its own unit error besides, so its null
result is superseded for the same reason.

**Superseded again, by work order 03: the statistic behind that
"confirmed" call took `|·|` before averaging over seeds, discarding
direction. The signed re-analysis of the SAME 90 runs (see "Work order
03, task 1") finds neither of the two growing points clears its own
threshold, and the two do not even agree on which arm is stickier — one
grows more anti-H4 with exposure, the other flips INTO the pro-H4
direction. Current verdict: H4's asymmetry claim is REJECTED** — not
merely "not yet confirmed": the direction is inconsistent across design
points, and H4 is a directional hypothesis. See "Work order 03, task 1"
for the numbers this rests on.

**Dataset provenance, one table for the whole experiment (standing
requirement, not previously consolidated — each number above cites its own
CSV inline, but no single place lists all six together).**

| dataset | substrate (`sbm_mirror_p`) | mechanism | background | rows | canonical for |
|---|---|---|---|---|---|
| `wave_a.csv` | 0.0 (pre-V7.6, reciprocity under-band) | `camp` | 0.7 | 135 | Wave A's own sign/consistency table only; superseded by `wave_a_reinstrumented.csv` for any magnitude, SESOI, or per-cross-contact claim |
| `wave_a_reinstrumented.csv` | 0.15 (V7.6) | `camp` (forced, to match Wave A's original mechanism) | 0.7 | 135 | all SESOI/magnitude/dosage/k-pinned-vs-stable checks laid onto Wave A's design; NOT byte-identical to `wave_a.csv` (substrate differs) |
| `wave_a_prime.csv` | 0.15 (V7.6) | `distance` (V7.3) | 0.5 | 135 | the entire Wave A′ section |
| `wave_b.csv` | 0.15 (V7.6) | `distance` | LHS around 3 named scenario centers, not one value | 216 | Wave B (H3b, diversity floor, response surface) |
| `wave_c.csv` | 0.15 (V7.6) | `distance` | the 3 largest-|effect| `engagement`/`distance` points from `wave_b.csv` | 15 | Wave C harm-side hysteresis |
| `wave_c_composition.csv` | 0.15 (V7.6) | `distance` | the 3 largest-|effect| `composition` points from `wave_b.csv` | 15 | Wave C benefit-side hysteresis (H4 asymmetry check) |
| `wave_a_prime_recal.csv` | 0.15 (V7.6) | `distance`, `affect_d0_mode="calibrated"` (work-order-01) | 0.5 | 135 | Wave A′ re-run at the recalibrated `d0` (population median, per-population — not one fixed value across rows); supersedes `wave_a_prime.csv` for any claim this section's own before/after comparison covers, which stays as the pre-recalibration reference |
| `wave_b_recal.csv` | 0.15 (V7.6) | `distance`, `affect_d0_mode="calibrated"` | LHS around 3 named scenario centers | 216 | Wave B re-run at the recalibrated `d0`; same relationship to `wave_b.csv` as above |
| `wave_c_recal.csv` | 0.15 (V7.6) | `distance`, `affect_d0_mode="calibrated"` | re-selected from `wave_b_recal.csv` — checked directly, the SAME 3 points as `wave_c.csv` (`select_hysteresis_points`'s top-3-by-magnitude ranking for `engagement` did not change, only the magnitudes did) | 15 | Wave C harm-side hysteresis, recalibrated |
| `wave_c_composition_recal.csv` | 0.15 (V7.6) | `distance`, `affect_d0_mode="calibrated"` | re-selected from `wave_b_recal.csv` — checked directly, NOT the same 3 points as `wave_c_composition.csv`: 2 of 3 match (`cross_cut` LHS 4/`distance`, LHS 0/`camp_pair`), the third changed from `cross_cut` LHS 4/`camp_pair` to `cross_cut` LHS 3/`distance` — the ranking is close enough for recalibration to reorder it, stated rather than assumed unchanged | 15 | Wave C benefit-side hysteresis, recalibrated |
| `wave_c_largergap.csv` | 0.15 (V7.6) | `distance` | the SAME 3 points as `wave_c.csv` (re-selected from the ORIGINAL `wave_b.csv`), `n_ticks_pre_withdrawal=1000` (was 60) | 15 | Decision 2's harm-side larger-gap hysteresis check (H4) |
| `wave_c_composition_largergap.csv` | 0.15 (V7.6) | `distance`, `camp_pair` | the SAME 3 points as `wave_c_composition.csv` (re-selected from the ORIGINAL `wave_b.csv`), `n_ticks_pre_withdrawal=1000` (was 60) | 15 | Decision 2's benefit-side larger-gap hysteresis check (H4) |
| `wave_c_largergap_ou_fit.csv` | n/a (derived) | n/a | per-seed 2-state OU refit (RMSE, R², peak_gap) over the 6 points in the two datasets above | 6 | Decision 2d's C1 residual-growth check |
| `wave_c_gap_ladder.csv` | 0.15 (V7.6) | `distance`, `affect_d0_mode="calibrated"` | 3 points selected JOINTLY from `wave_b_recal.csv` by both arms' mean \|Δ\| (work order 02, task 1) — a common design, not per-arm selection | 90 | Work order 02's H4 asymmetry re-test (60/300/1000-tick pre-withdrawal ladder, both arms at every point) |
| `wave_c_gap_ladder_ou_fit.csv` | n/a (derived) | n/a | per-seed 2-state OU refit over all 90 rows in the row above | 18 | Work order 02's secondary RMSE/peak_gap check, all 3 rungs |
| `wave_c_gap_ladder_dbs_split.csv` | n/a (derived) | n/a | per-seed 2-state OU refit over the SAME 90 rows as `wave_c_gap_ladder.csv`, keeping `dBs`/`delta_peak` per seed instead of only the pooled RMSE | 90 | Work order 03, task 2's `ΔBs`/`Δ_withdrawal` split-ratio decomposition |
| `wave_a_prime_recal_cross_contact_fraction.csv` | 0.15 (V7.6) | `distance`, `affect_d0_mode="calibrated"` | `wave_a_prime_recal`'s own 45-point design, all 4 arm-values audited directly | 180 | Work order 02, task 6's dosage-confound closure (A4, recalibrated classifier) |

**Two standing items this project defers rather than silently drops.**
H1's falsifier has now been searched for across Wave A (one background,
0.7), Wave A′ (one background, 0.5, full dial range), and Wave B (3
scenario neighborhoods) — and DOES fire, once, narrowly (`composition`
positive at `consolidated_two_camp`'s own most extreme LHS corner under
`distance` targeting, found on a later pass through data this project
already had — Wave B section above). One confirmed exception at one
scenario's own corner does not settle whether OTHER crossings exist
elsewhere in the cube — deferred three times to "a denser design" without
ever stating what would actually settle THAT broader question. Stated
now: a real Morris elementary-effects design needs multiple (a handful,
e.g. 4-6) random trajectories through the full 3-dial cube, each
perturbing one dial at a time from a random base point — order 20-40 design
points chosen to SPREAD across the cube rather than cluster around named
scenarios, specifically so a fitted surface has enough effective
neighborhoods to validate out of sample (the D1/D2 gap just found in the
response surface above). Anything smaller repeats this session's own
under-determined-fit problem. Separately: Wave C's point selection
(`select_hysteresis_points`, top-3 by `|delta_aff_plateau|`) picks from
Wave B's 2-seed ESTIMATES, not from the true per-point effect size — with
only 2 seeds, a point's estimate can rank in the top 3 partly by which way
its own sampling noise happened to land, so "the 3 largest effects" this
section's own language uses at first mention should be read as "the 3
largest 2-seed ESTIMATES," a real but weaker claim than it sounds.

**Scope, same discipline as Wave A/A′.** 2 seeds × 6 LHS points per
scenario here vs. the brief's own §5.3 default (~30 points, 10 seeds) — a
~1/20 reduction matching Wave A's own precedent, stated as such in
`run_wave_b`'s own docstring. No response-surface regression fit over the
LHS points (the points are collected; fitting one is a separate,
not-yet-built analysis step). No calibration against a real corpus (SS8),
still out of scope.

## Pre-submission checklist run against Experiment 03

Run against the full Experiment 03 arc above (Wave A through Wave C) per
the project's own checklist (`findings-pre-submission-checklist.md`;
Review 01/02 + brief §7). Two items found real, load-bearing problems
rather than wording gaps on the first pass: D1 (the response-surface
R²=0.91 was ~93% scenario-identity, not a validated surface) and A4/B3
(the per-cross-contact dosage check does not discriminate at this
substrate — reversed from "resolved" to "open").

**Second pass, after the checklist's own amendments.** The first pass's
table (below) scored B1, B4, and C1 **pass** on reasoning rather than a
fresh measurement — exactly the failure mode the checklist's own amended
scoring-discipline section now names, applied here to itself rather than
just added to the document. Re-run properly:

- **B1** was scored pass because `affect_d0`'s D=3 mismatch was "already
  flagged" — flagged is not measured. Actually measured this time: joined
  every `distance`-mode engagement against the engaging user's and post
  author's stable pre-period camp label and computed `phi(d)` by TRUE
  camp pairing, not just its aggregate level. Result (corrected once more
  after work-order-01 caught this measurement's own RMS-units bug — see
  the Wave A section above) is more precise than "saturates alike": `phi`
  DOES discriminate real camp membership (mean 0.543 same-camp vs. 0.742
  cross-camp at background=0.7, widening with tribalization), and same-camp
  contact is essentially NEVER near-saturated (phi>0.8 in 0.09% of it) —
  the discrimination is real, but it lives in the same distance range
  `affect_d0=1.0` already sits in, which is why the old threshold still
  fails to separate the two classes cleanly. Now **fixed**, not pass.
- **B4** was scored pass on "no new claim was made this pass" — the
  amended B1 clarification's "no new constant is not the test" applies
  identically here. Actually re-run: `grep -rn "camps is not
  None\|camps is None" discourse_lab/` finds a THIRD site
  (`metrics/dmp.py`), not two. Checked whether it matters: `dmp_table`/
  `dmp_thread_count` are never imported by Experiment 03's driver or
  notebook — confirmed dead code with respect to every claim in this
  document. Now **fixed** (re-enumerated, found and cleared a real gap),
  not pass-by-assumption.
- **C1** is left at **pass**, but on a tighter basis than before: its
  measurement (the OU 2-state fit, R²>0.999 at all 3 points) was real,
  freshly-computed analysis done earlier in this same investigation, not
  reasoning about a flag — the distinction the amendment draws is between
  measurement and reasoning, not between this commit and an earlier one.
  Stated explicitly here so the "pass" is not resting on the same
  unstated assumption B1 and B4 were.
- **C3** was reasoned as pass ("pre-existing"); re-verified fresh this
  pass instead by reading `ARMS` directly (`experiment03_bubble_
  intervention.py`): `engagement` sets `kernel_theta` only, `composition`
  sets the SAME `kernel_theta` plus `valence_gamma0` — confirmed, not
  assumed. Pass stands, now on a fresh check.
- **C5** (new item, added after the first pass): `diversity_floor_break_
  even`'s 0.997-1.024 flat band is checked against a sensitivity test —
  `none`'s own version of the same ratio is equally flat (dispersion is
  genuinely more stable here than animus/ideology, not evidence the check
  is broken), but netting it out surfaces a small, consistently positive
  effect (+0.001 to +0.003) invisible in the raw statistic. **Fixed** —
  the "close to orthogonal" conclusion is revised to rest on a checked
  small net effect rather than an unchecked flat ratio.

| item | result | one line |
|---|---|---|
| A1 — sign claims paired with magnitude + SESOI [blocking] | fixed, one stated limitation | Original Wave A and Wave A′ H2 now carry full SESOI tables; Wave A′ H3/Wave B H3b's `delta_aff_plateau` magnitude claims still lack one (`delta_aff_level_none` does not exist in `wave_a_prime.csv`/`wave_b.csv` — would need a re-run to close) |
| A2 — exclude degenerate cells from counts | pass | Already excluded/labeled at every site checked (ideological=0.1 gate, engagement's own kernel_theta gate) |
| A3 — seeds are not independent draws | fixed | Standing caveat added where seed-cell counts first appear |
| A4 — per-contact denominator's selection fraction characterized | fixed, reversed a prior claim | Audited directly (45 points × 4 arms): `d_cross` selects 80-85% of ALL contact almost everywhere — not restrictive. The "dosage confound resolved" claim from the previous commit was wrong; rewritten to "open, untested". Closed, not just re-opened: work order 01 recalibrated `d_cross` to the realized camp boundary, and work order 02's re-audit on that classifier finds 42-46% selected (genuinely restrictive, varying by point) — `engagement`'s backfire sign survives 40/40 |
| A5 — each comparison against its own matched reference | fixed | H2/SESOI tables recomputed as median-of-per-point-ratios, not ratio-of-pooled-medians; changed two conclusions (engagement's H2 contribution, exposure's SESOI uniformity) |
| B1 — new/reused constants measured in their actual regime [blocking] | fixed (was wrongly scored pass) | `phi(d)` measured directly by true camp pairing: discriminates real membership but both classes saturate because `affect_d0=1.0` sits well below the true same-camp distance scale (~2.3) |
| B2 — thresholds reported as realized distributions | pass | Bimodality gate, k pinned/stable, cross-contact selection fraction, and (this pass) `phi` by true camp pairing all reported as measured distributions |
| B3 — shared constant checked in each consumer | fixed | `d_cross`/`affect_d0` checked directly in the cross-contact classifier; B1's fresh measurement now separately checks the affect-drive consumer too |
| B4 — every consumer of a gated feature enumerated | fixed (was wrongly scored pass) | Fresh `grep` finds a third site (`metrics/dmp.py`); confirmed unused by Experiment 03, so the previously-known two remain the complete relevant set |
| C1 — a derivation must contain the mechanism it predicts about [blocking] | pass | H4's OU derivation contains the sign-symmetry property itself; measurement was real (R²>0.999) though done earlier in this investigation, not reasoning about a flag |
| C2 — "not supported" vs. "not testable" distinguished | pass | H5 (k pinned vs. stable) and now A4 (untested vs. negatively tested) both apply the distinction |
| C3 — an arm's own construction separated from its attributed effect | pass | Re-verified fresh this pass by reading `ARMS` directly: `engagement`/`composition` share `kernel_theta`, differ only in `composition`'s added `gamma0` |
| C4 — alternative explanation stated before naming a finding | pass | No new unvalidated name asserted; response-surface downgrade explicitly avoided over-naming what the LOO check does not support |
| C5 — a null statistic gets a sensitivity check | fixed | `diversity_floor_break_even`'s flat 0.997-1.024 band checked against `none`'s own (also flat) version; netted out over the SAME 90-row domain the statistic is itself defined on, reveals a small real effect for `composition` (+0.0028) the raw ratio hid, ~zero for `exposure`, undefined for `engagement` |
| D1 — fits report effective design points, not row count [blocking] | fixed | `camp_pair`'s 18 points are 3 effective neighborhoods against 4 parameters — under-determined; stated explicitly with cluster-only R²=0.844 and leave-one-out R²=0.072 |
| D2 — fits validated out of sample or with clustering absorbed | fixed | Leave-one-scenario-out and scenario-fixed-effects both added for `camp_pair` and `distance` |
| D3 — algebraically dependent quantities counted once | pass (fixed earlier this session) | `toward_own_pole` reported as primary; `toward_mean`/`toward_other_camp` noted as near-mirror-images, not counted as independent confirmations |
| D4 — good and bad fits held to the same standard | fixed | Leave-one-out applied to both `camp_pair` (R²=0.072) and `distance` (R²=-8.95), not only to the fit already read as a negative result |
| dataset provenance | fixed | New table: substrate, mechanism, background, canonical-for, all six CSVs |
| stopping rules before the next wave | fixed | H1's own concrete Morris-design coverage stated (order 20-40 space-filling points), after three waves of deferring it without saying so |
| selection on noisy estimates labelled | fixed | Wave C's point selection now explicitly named as "largest 2-seed ESTIMATES," not "largest effects" |
| house virtues (record gap first, test first, default off, non-comparability stated) | pass | Consistent throughout; this pass's own new datasets follow the same pattern |

## Work order 01 — decisions made, breaking the affect_d0/d_cross tie

Two decisions from a follow-up review round (referred to throughout as
"work-order-01"), plus its own acceptance criteria, which supersede parts
of the B1 write-up two commits above (both this section and that one now
use the corrected, RMS-aware numbers — the acceptance criteria caught a
real units bug in the underlying measurement before any code was written
against it, which is exactly what running acceptance criteria in advance
is for).

**Acceptance criterion A, computed before any code changed: does
recalibrating `d0` alone even reach what it was chosen for?** With
`phi(d) = d/(d+d0)` and REALIZED (not class-mean-approximated) same/cross
distance distributions, the same/cross `phi` ratio `R(d0) =
mean(phi(d_same))/mean(phi(d_cross))` decreases monotonically from 1 at
`d0=0` to a floor as `d0→∞` — and that floor is `mean(d_same)/
mean(d_cross)`, unchanged by ANY choice of `d0` because both numerator and
denominator scale together. Computed directly (not approximated from
class means) at the three backgrounds this project already had numbers
for:

| background | `d0=1.0` ratio | `d0`≈population-median ratio | floor (`d0→∞`) |
|---|---|---|---|
| 0.5 | 0.776 | 0.711 (at `d0`=1.855) | 0.538 |
| 0.7 | 0.732 | 0.630 (at `d0`=2.195) | 0.433 |
| 0.9 | 0.708 | 0.572 (at `d0`=2.571) | 0.366 |

Population-median calibration moves the ratio roughly a third of the way
from `d0=1.0` toward the floor (e.g. at background=0.7: 0.732→0.630, 34%
of the 0.732→0.433 gap) — a real, meaningful improvement, but nowhere near
`"camp"` mode's ratio of 0.00, and no calibration of `d0` alone can get
there: the floor is a property of the class MEANS, which `d0` does not
touch. **This does not invalidate breaking the tie** (`d_cross` still
needs its own calibration, below) **— it means the phi-gain recalibration
should be adopted knowing what it delivers.** Of the three options this
criterion named (accept the ceiling and reframe the grounding; redesign
`phi`'s functional form so same-camp gain can approach zero; keep `"camp"`
as the default above the gate and reserve `"distance"` for below it, where
`"camp"` cannot run at all) — **this project takes option 1: recalibrate
`d0` to the population median, keep it, and state the affect channel's
grounding precisely** (below). Option 3 would revert most of Wave
A′/B/C's own above-gate results to a different mechanism than the one they
were run and reported under, which decision 1's own re-run plan (below) is
not built for and which this pass has no budget to redo from scratch.
Option 2 is a new mechanism needing its own theoretical warrant and
conformance tests, out of scope here. `d0`'s own realized value at the
three backgrounds (1.855 / 2.195 / 2.571) is itself evidence for taking
option 1 rather than deferring the whole decision: it is not a wild
outlier from `d_cross`'s own realized boundary at the same backgrounds
(1.98 / 2.40 / 2.84, `_camp_boundary_d_cross`) — both land in the same
neighborhood, well above the old constant of 1.0, because (an unanticipated
finding of this same measurement) `stance_polarization` differentiates
only axis 0 of D=3, so two undifferentiated noise axes dominate total
distance for both statistics almost equally. **Grounding restated
precisely, per this criterion's own request:** this project's `"distance"`
affect channel, even after recalibration, is distance-graded escalation
with an out-group PREMIUM (same-camp contact still contributes ~60-71% of
what cross-camp contact does to the animus update, not close to zero) —
`drift.py`'s own Rathje citation describes an out-group-SPECIFIC
mechanism, which this is not, at any `d0`.

**Work order 02, task 4 restated this in code, not just here; work order
03, task 3 confirmed it landed.** `drift.py::affect_delta`'s module
docstring (lines 385-387) and `MODEL.md`'s C1 mechanism-table row (line
989) both carry the same "distance-graded escalation with an out-group
PREMIUM, not the out-group-SPECIFIC mechanism the Rathje citation itself
reports" language — checked directly against both files, not assumed
from memory of having written it. No fix was needed.

**Decision 2, pre-registered: the larger-gap hysteresis check for H4.**
The current 2-state OU fit (R² > 0.999 at the 5-seed MEAN trajectory,
peak gaps ~1e-2) shows the endogenous-valence feedback term is small at
those magnitudes; it does not show the term is absent, because the
derivation predicting equal recovery is the same derivation that drops
it. Settling this needs peak gaps roughly an order of magnitude larger
(~1e-1) — reached by lengthening the pre-withdrawal window, not
strengthening `kernel_theta` (that would change what the arm IS, not how
long it has been accumulating, and would make the withdrawal comparison
dirtier). Written down BEFORE running the new points, per this project's
own C1/C2 (acceptance-criteria doc for this work order):

- **C1's threshold.** Refit the EXISTING 6 Wave C points (3 `engagement`/
  harm, 3 `composition`/benefit) per SEED rather than on the 5-seed mean
  (the mean-trajectory fit is what FINDINGS.md's Wave C section already
  reports; this is the same fit run 5 times per point instead of once, to
  read across-seed spread). Per-seed R² ranges 0.982-0.999; mean RMSE
  0.000156, pooled SD of per-seed RMSE across all 6 points 0.000053 (the
  noise floor). **Growth is confirmed only if the new ~1e-1-gap points'
  mean per-seed RMSE exceeds 0.000156 by more than 3× that noise floor
  (> 0.000315 in absolute terms) — a plain "the residual is bigger"
  is not enough, since RMSE naturally has more room to be large against a
  10x bigger gap even under a still-good fit.** Below that line, the
  residual is judged flat and the OU mechanism's own prediction stands
  unmodified at the new scale too.
- **C2's threshold.** Current point estimates: `composition` (benefit)
  recovery_fraction 0.6078 (SD 0.0270, n=15 seed-runs across 3 points),
  `engagement` (harm) 0.6250 (SD 0.0253, n=15) — a difference of 0.0172
  against a pooled seed-spread SE of 0.0087 (already only ~2x the SE,
  consistent with the existing "not supported" reading). **A real
  asymmetry at the new magnitude is confirmed only if
  `|recovery_fraction_engagement - recovery_fraction_composition|` exceeds
  3× the pooled SD (~0.081) at the new points — set before looking at
  them, matching C1's own 3-sigma-style bar rather than picking a
  post-hoc threshold that happens to make whatever the new numbers show
  look decisive.**
- **C3/C4, re-verified rather than assumed to survive the window
  change:** the schedule prefix bit-identity test
  (`test_schedule_gives_a_bit_identical_prefix_and_diverges_after`) does
  not depend on tick COUNT, only on the schedule mechanism itself, so it
  needs no new case for a longer pre-withdrawal window — confirmed by
  reading the test, not re-asserted with a new fixture, since it already
  parameterizes the fork tick rather than hardcoding one. `ARMS`
  (`experiment03_bubble_intervention.py`) is read fresh, not assumed:
  `engagement`/`composition` still differ only in `composition`'s added
  `valence_gamma0` — unchanged by this decision, which touches
  `n_ticks_pre_withdrawal` only, never the arm definitions.

## Decision 1d — the recalibrated re-run, and what actually moved

Wave A′, Wave B, Wave C (harm), and Wave C (benefit) re-run at
`affect_d0_mode="calibrated"` (2822s total, no errors; datasets in the
provenance table above). Reported as before/after pairs per the work
order's own instruction, not as silent replacements.

**Headline: almost nothing that mattered changed sign or qualitative
shape. What moved was magnitude, by a modest and consistent amount, and
one genuinely new finding surfaced that predates this recalibration
entirely.**

- **Wave A′'s SESOI ratios on `toward_own_pole`** (the H2-analog):
  22.7%/24.1%/20.9% (composition/engagement/exposure, OLD) →
  18.9%/25.5%/14.2% (NEW). All six numbers clear the brief's 10% SESOI
  comfortably both before and after; composition and exposure moved down,
  engagement up, none crossed the line in either direction. **New this
  pass, because `delta_aff_level_none` did not exist when Wave A′
  originally ran:** a full SESOI check on `delta_aff_plateau` itself —
  `composition` clears overwhelmingly (19.8x median, 40/40 points),
  `engagement` clears well (2.8x median, 39/40), but **`exposure`'s
  median ratio is 0.81 — BELOW 1.0, and only 13/40 points individually
  clear it.** `exposure`'s animus effect, over the full dial range, is
  mostly indistinguishable from `none`'s own drift by the brief's own
  rule — a real finding this section could not make before, not a
  consequence of recalibration.
- **The camp-aware-targeting finding (carried-over item 1's own open
  question) is RESOLVED, not just carried forward again.** At the
  identical gated design point (ideological=0.1), `engagement` is STILL
  an EXACT 0.0 on every measure across all 5 seeds under the recalibrated
  mechanism — bit-for-bit the same null result as before recalibration —
  while `composition` is STILL measurable and consistent (-0.00509 to
  -0.00535 across seeds, essentially the same magnitude as pre-
  recalibration's -0.0046 to -0.0060). Recalibrating `affect_d0`/`d_cross`
  touches the AFFECT channel and the analysis-only classifier; it does
  not touch `exposure/kernel.py::compute_features`'s `CONDITIONAL_
  FEATURES` gate on `outgroup`, which is what actually freezes
  `engagement`'s `kernel_theta` override below the bimodality gate. Since
  changing the affect-channel calibration changed NOTHING about
  `engagement`'s below-gate behavior, the "near-uniform gain" competing
  explanation is ruled out by construction, not just by assumption: if
  that had been the mechanism, recalibrating the gain would have moved
  it. **The camp-aware-targeting name stands, now on separated evidence
  rather than an unruled-out competitor.**
- **A genuinely new finding, NOT caused by recalibration, surfaced by
  doing this comparison at all: H1's falsifier fires once.** Reported
  above (Wave B section) and in the Verdict table (notebook) — `wave_b.
  csv`, the ORIGINAL pre-recalibration dataset, already contains 2
  positive `composition` rows out of 216 (both seeds of `consolidated_
  two_camp` LHS=1, `distance` targeting), missed by every previous pass
  through this document including this section's own earlier "Zero sign
  flips" claim. The recalibrated data has the same 2 positive rows at the
  identical point, smaller in magnitude (0.0074-0.0085 vs. 0.0099-0.0126)
  — confirming the finding is real and substrate-independent, not an
  artifact of either calibration. **SESOI-checked (work order 02, task
  3), not just sign-checked: both seeds clear the brief's own 10% line by
  3.8x/4.8x** — a real effect, not one dismissible as below this
  project's smallest effect of interest, though resting on only 2 seeds
  at this specific point (the cross-calibration reproduction above is
  what carries this claim, not seed count).
- **The dosage-confound check for `engagement` — CLOSED (work order 02,
  task 6): NOT a dosage artifact.** Left OPEN in the pre-submission-
  checklist pass because `d_cross`'s classification did not discriminate
  well at ANY calibration tried there (A4: 80-85% of ALL contact
  selected almost everywhere). Re-audited on the recalibrated classifier,
  same shape as A4 (45 points x 4 arm-values, `wave_a_prime_recal`'s own
  design, `n_users=1000`, `n_ticks_total=160`,
  `results/experiment03/wave_a_prime_recal_cross_contact_fraction.csv`,
  874s): the cross-contact selection fraction now averages **42-46%**
  across arms (median 42.6-46.0%, range 22.5-58.4%) — genuinely
  restrictive and, unlike A4's own finding, varying substantially by
  design point rather than pinned near one value. `engagement`'s backfire
  sign survives intact under this now-informative denominator: **40/40
  measurable points positive** (5 below-gate, same as `delta_aff_plateau`
  and `delta_aff_per_contact`'s own counts). More precisely: the
  `per_cross_contact`/`per_contact` RATIO is not a constant rescale (which
  would mean the restriction adds no information) — median 2.17x, range
  1.71x-3.89x, tracking each point's own `1/cross_contact_fraction`
  almost exactly, confirming the denominator is doing real per-point work.
  **Revised claim: `engagement`'s hostility-raising backfire is not an
  artifact of measuring more contact overall — isolating cross-camp
  contact specifically does not shrink or reverse it, and its per-event
  rate is if anything LARGER once diluting same-camp contact is removed.**
- **Response-surface coefficients (`camp_pair`, `composition`).**
  affective -0.0170→-0.0121, ideological -0.0102→-0.0068, structural
  +0.0084→+0.0072, R²=0.908→0.883 (similar), but **leave-one-scenario-out
  R² = 0.072→-0.455 — WORSE, not better.** This is independent
  replication of the D1 finding from the pre-submission-checklist pass:
  the fit's headline R² is stable-looking, but its actual out-of-sample
  behavior is unstable across two different calibrations of the same
  under-determined design (3 effective neighborhoods, 4 parameters) —
  exactly what D1 predicts an under-determined fit will do, not a
  coincidence specific to one dataset. `distance` mode: R²=0.415→0.403,
  LOO R²=-8.99→-9.26 (already a clear negative result, now more so).
- **H3b (`engagement`, `camp_pair` vs. `distance`), B1's own n-count
  check:** 24/24 measurable cells both before and after (the 12
  `low_tribalization` cells stay an exact structural 0.0 both times) —
  this comparison is orthogonal to the affect-channel recalibration by
  construction (it is about the KERNEL's `outgroup`/`cross_distance`
  features under different `targeting_mode`s, not the animus-update
  mechanism), so an unchanged count here is confirmatory, not
  uninformative.
- **Wave C (H4): recalibration barely moves these numbers either way —
  the asymmetry finding itself is superseded by work order 02, below,
  and work order 02's own "confirmed" call is in turn superseded by
  work order 03's signed re-analysis to REJECTED (see "Work order 03,
  task 1") — neither is about this recalibration.** Harm recovery_fraction
  0.6250→0.6298 (SD 0.0253→0.0206), benefit 0.6078→0.6157 (SD
  0.0270→0.0198) — both sides moved together, by less than their own
  seed spread, keeping the harm/benefit difference small in both
  datasets AT THIS (per-arm-selected, pooled) design; work order 02's
  common-design, per-point re-test later found a real asymmetry the
  pooling here could not see. Peak gaps shrank somewhat (e.g. the largest point:
  0.0356→0.0240) — recalibration lowers `engagement`/`composition`'s
  overall animus impact a little, consistent with the response-surface
  coefficients shrinking too, but does not touch the recovery MECHANISM
  (OU reversion), so `recovery_fraction` itself barely moves.
- **B3, checked not assumed: `agree_delta` is provably unchanged.**
  `dynamics/tick.py`'s `agree_delta` computation was not edited by V8 at
  all — `affect_d0_calibrated` is a new line added immediately after it,
  reading its already-computed value, not a modification to how it is
  computed. The full test suite (400 passed, 2 xfailed, 1 xpassed,
  including V1's endogenous-valence tests that exercise `agree_delta`
  directly) passing with zero regressions is the actual check, not an
  assumption resting on the diff being small.
- **B5: H1's falsifier record does not just "reset" — it resolves to a
  confirmed, narrow exception**, true under both the old and new
  calibration (above). The Verdict table's H1 row is corrected
  accordingly (notebook, same pass as this finding, per this project's
  own amended checklist).

**What this section does NOT do.** No fresh per-cross-contact dosage
audit on the recalibrated data (open, above). No re-run of Wave A
(original, `affect_drive="camp"`) or `wave_a_reinstrumented.csv` — both
stay on `affect_d0_mode` irrelevant to them (`"camp"` mode never reads
`affect_d0` for its own gain; the provenance table already marks them as
historical baselines). No attempt to re-derive a response surface that
would actually validate out of sample — decision 1d's own finding is that
this fit is unstable across calibrations, which is evidence AGAINST
spending more effort on the current 18-point design rather than a reason
to keep refitting it.

## Decision 2d — the larger-gap hysteresis check, run and resolved

Both sides re-run at `n_ticks_burn_in=60, n_ticks_pre_withdrawal=1000,
n_ticks_post_withdrawal=60` (previously 60/60/60), same `n_users=1000`,
same 5 seeds. The 3 largest-|effect| points per side are re-selected from
the ORIGINAL, pre-recalibration `wave_b.csv` — deliberately the SAME
points `wave_c.csv`/`wave_c_composition.csv` already used, not
`wave_b_recal.csv`'s, so this check's own before/after comparison (below)
stays against the exact short-gap baseline C1/C2 were pre-registered
against, not a recalibrated one. `wave_c_largergap.csv` (engagement/harm,
15 rows, 1356s) and `wave_c_composition_largergap.csv`
(composition/benefit, 15 rows, 645s — re-run once after an unrelated
mid-job cache-eviction crash; both datasets reflect a complete, clean
run) plus `wave_c_largergap_ou_fit.csv` (the per-seed OU refit this
section reports).

**Peak gaps actually reached: 3.5-4.4x larger, not a literal order of
magnitude, but landing in the targeted ~1e-1 range regardless.**

| point | old peak_gap (60-tick) | new peak_gap (1000-tick) | ratio |
|---|---|---|---|
| engagement `consolidated_two_camp` LHS1 | 0.0235 | 0.1039 | 4.42x |
| engagement `consolidated_two_camp` LHS5 | 0.0142 | 0.0567 | 4.01x |
| engagement `cross_cut` LHS0 | 0.0124 | 0.0473 | 3.81x |
| composition `cross_cut` LHS4 (`camp_pair`) | -0.0097 | -0.0371 | 3.83x |
| composition `cross_cut` LHS4 (`distance`) | -0.0103 | -0.0371 | 3.61x |
| composition `cross_cut` LHS0 (`camp_pair`) | -0.0104 | -0.0365 | 3.50x |

16.7x more pre-withdrawal ticks bought 3.5-4.4x more peak gap —
accumulation is sub-linear in exposure time, consistent with a
mean-reverting mechanism where `k*(X-Bs)` increasingly opposes further
growth in `X` as `X` itself grows.

**C1 (OU-fit residual growth): NOT confirmed — the residual stays flat.**
Per-seed refit of the SAME 2-state model (`M = [[1-k,k],[k_b,1-k_b]]`,
least-squares on `ΔBs` alone, `Δ_withdrawal` fixed from the observed
trajectory) on all 6 new points: pooled mean RMSE **0.000181**, against
baseline 0.000156 and the pre-registered threshold of >0.000315 (baseline
plus 3x its own 0.000053 noise floor). Per-seed R² 0.962-0.999 (baseline
0.982-0.999) — comparable fit quality at ~4x the gap magnitude, not
degrading. The OU model's prediction stands at the new scale too: this is
the same linear mechanism describing a larger gap, not a fit that happens
to still work by coincidence.

**Same comparison, in units that do not depend on the gap magnitude
(work order 02, task 2).** The 0.000181-vs-0.000156 comparison above is
absolute RMSE across points whose peak gaps differ 3.5-4.4x — informative
for the pre-registered threshold (which was itself set in absolute units
deliberately, unlike C2's), but not the fairest single number for the
"is the residual growing" question on its own. `RMSE / |peak_gap|`, mean
of the 6 per-point ratios: **0.01175 at the short gap, 0.00355 at the
larger one — a 3.3x DECLINE**, not growth. A dropped term quadratic in the
gap would raise absolute RMSE ~16x for a 4x gap increase; it rose 1.16x,
and the normalized ratio falls. Both unit choices agree on the verdict;
the normalized one rules out "the absolute comparison happened to favor
no-growth because the gaps were also growing" as a competing explanation.

**C2 (harm/benefit asymmetry): NOT confirmed by this pre-registration's
OWN threshold — and that threshold turned out to be the wrong test
(work order 02, tasks 1 and 7).**
`composition` (benefit) recovery_fraction mean **0.2121** (SD 0.0104,
n=15); `engagement` (harm) mean **0.2339** (SD 0.0108, n=15); difference
**0.0218**, against baseline difference 0.0172 and the pre-registered
threshold of >0.081 (~3x the original pooled SD). The two distributions
still overlap almost completely — **but this threshold was fixed in
`recovery_fraction`'s SHORT-gap absolute units, and `recovery_fraction`'s
own SD fell ~2.5x at this larger gap (0.0270/0.0253 -> 0.0104/0.0108
above), so 0.081 was functioning as roughly an 8-sigma bar here, not the
3-sigma one it was set to be** — checked directly and named as this
project's own standing checklist amendment, not asserted. This section's
own verdict stands as the record of what THIS pre-registration found;
work order 02's own re-test, on a scale-relative statistic and a design
that also fixes the arm/point confound and cross-point pooling this
section's points still carry, found asymmetry confirmed. See "Work order
02, task 1" for the numbers.

**A real change neither threshold was built to catch, checked rather than
left as a loose end: recovery_fraction itself collapses at the new
magnitude** (harm 0.6250→0.2339, i.e. to 37% of its old value; benefit
0.6078→0.2121, to 35%). This is not a third result contradicting the
first two — it is the same sign-symmetric OU mechanism (above) doing
something the two pre-registered thresholds do not measure. Refitting
`ΔBs` (the slow-block gap already accumulated by the moment of
withdrawal) as a fraction of `Δ_withdrawal` at these same 6 new points:
**64-72% (pooled 67%)**, against the 5-9% the ORIGINAL short-gap Wave C
mechanism check reported at these same 3 harm points' short-gap
counterparts. A 1000-tick pre-withdrawal window gives the slow `Bs` block
(`k_b = k/10`, one-tenth the fast block's own reversion rate) far more
time to drift toward the elevated state before withdrawal, so by the time
withdrawal happens most of the total gap is stored in the SLOW component
— which the same fixed 60-tick post-withdrawal window recovers only a
small fraction of. This is the two-timescale mechanism interacting with
exposure DURATION, not a new mechanism, and it does not differ between
harm and benefit (both drop to within 2 points of the same 35-37%
fraction of their old recovery) — so it does not reopen C2, it explains
why recovery_fraction's absolute LEVEL is not itself a stable statistic
across exposure lengths the way C1/C2's own ratio-based thresholds are.

**Decision rule applied, per the pre-registration: neither threshold
fired, so H4's verdict is unchanged, now checked at a substantially
larger gap.** Partial, sign-symmetric persistence (the non-asymmetry half
of H4) continues to hold; the ASYMMETRY half continues to not hold. What
DOES change with exposure duration is the absolute fraction a
fixed-length withdrawal window recovers — a named property of the OU
mechanism's two timescales, not a reason to suspect the harm/benefit
comparison itself.

## Work order 02, task 1 — pre-registered: the gap-ladder H4 re-test on a common design

**Why this needs a second re-test, not a footnote on the first.** Decision
2d's C2 threshold (0.081) was 3x the SHORT-gap pooled SD of
`recovery_fraction`. At the larger gap, `recovery_fraction`'s own SD fell
~2.5x alongside its mean (composition 0.0270→0.0104, engagement
0.0253→0.0108) — so the SAME absolute threshold, applied there, was
functioning as roughly an 8-sigma bar, not the 3-sigma one it was set to
be. The pre-registration stands as recorded and its verdict is not being
retro-fitted; this section registers a DIFFERENT, scale-relative test
before running anything new. Two more defects, both older than the
larger-gap run: harm and benefit were measured at DIFFERENT design points
(only `cross_cut` LHS0 overlapped), confounding arm with point; and each
side's 15 rows are 3 points x 5 seeds, not 15 independent draws, so
pooling across points overstates precision. This section's own design
fixes both.

**Points, selected before running, from `wave_b_recal.csv`.** Ranked by
the mean of `engagement`'s and `composition`'s own `|delta_aff_plateau|`
(each averaged over its 2 Wave B seeds first) at every
(`scenario`, `lhs_index`, `targeting_mode`) triple where NEITHER arm is
degenerate (`low_tribalization` x `camp_pair` is the one family where
`engagement` is structurally 0.0 at every LHS index — excluded, though it
was never close to the top of this ranking anyway). Top 3, all under
`distance` targeting:

| point | composition \|Δ\| | engagement \|Δ\| | mean | affective | ideological | structural |
|---|---|---|---|---|---|---|
| `consolidated_two_camp` LHS1 | 0.00792 | 0.03082 | 0.01937 | 0.9464 | 0.8849 | 0.7319 |
| `cross_cut` LHS0 | 0.01146 | 0.01558 | 0.01352 | 0.6714 | 0.5867 | 0.0126 |
| `cross_cut` LHS4 | 0.01395 | 0.00964 | 0.01180 | 0.5441 | 0.8641 | 0.2978 |

4th place (`cross_cut` LHS5, mean 0.01143) trails LHS4 by only ~3% —
close, stated rather than silently assumed decisive. These are **"largest
2-seed estimates," not "largest effects"**: Wave B ran 2 seeds per point,
so this ranking is itself a noisy estimate, named as one. All 3 points
happen to already appear in work order 01's own point lists (LHS1 and
LHS0 on the harm side, LHS4 on the benefit side) — continuity, not a
selection artifact, since the ranking here was computed fresh from both
arms jointly.

**Gap ladder.** `n_ticks_pre_withdrawal` in {60, 300, 1000}; burn-in and
post-withdrawal fixed at 60 (`n_ticks_total` = 180, 420, 1120), `targeting_
mode="distance"` throughout (the mode all 3 points were selected under),
`n_users=1000`, seeds 0-4. Both arms run at all 3 points x 3 windows: 3 x
2 x 3 x 5 = 90 arm-runs. The matched `none` run at a given (point, window,
seed) does not depend on `arm` (`run_hysteresis`'s `none` call takes no
targeting-mode or arm argument), so the two arms' `none` twins are the
SAME config and only 3 x 3 x 5 = 45 of them are ever actually computed,
reused (cache hit) for the second arm. The existing 60-tick and 1000-tick
data (`wave_c.csv`/`wave_c_composition.csv`/`wave_c_largergap*.csv`) are
at the OLD, per-arm-selected points, not these — all three rungs here are
fresh runs, none reused. Budget: the 1000-tick rung is the dominant cost;
scaling from work order 01's measured 1356s for 15 rows (3 points x 5
seeds x 1 arm, ALSO paying for 15 unique `none` runs) to this rung's 45
unique arm-runs plus 15 unique `none` runs (60 total unique sims vs.
that run's 30), a rough estimate is 60-100 minutes for the 1000-tick rung
alone, with 300-tick and 60-tick scaling down roughly with tick count
(~40% and ~17% of that respectively) — actual total to be reported
against this estimate, not silently substituted for it.

**Primary statistic — written down before looking at any new number.**
Per point `p`, window `w`, seed `s` (seed-PAIRED between arms: seed `s`'s
`engagement` run and seed `s`'s `composition` run share the same burn-in
population realization, differing only in which arm is applied, so
pairing by seed index cancels population-level noise the way an unpaired
comparison cannot):

    A(p, w, s) = |rec_engagement(p,w,s) - rec_composition(p,w,s)| / mean(rec_engagement(p,w,s), rec_composition(p,w,s))
    A(p, w)    = mean over s of A(p, w, s)            [point estimate, n=5]
    SD(p, w)   = sample SD (ddof=1) over s of A(p, w, s)   [uncertainty]

computed WITHIN each point, never pooled across the 3 points (that pooling
is exactly A3's own objection). This is one specific, defensible reading
of the work order's "per-seed values give A's across-seed SD" instruction,
stated explicitly because the alternative (a ratio-of-seed-MEANS point
estimate with a separately-estimated SD) was also consistent with that
wording and the two are not identical — this project takes the mean-of-
paired-ratios convention throughout this section.

**Asymmetry confirmed only if BOTH hold:**
- `A(p, 60) < A(p, 300) < A(p, 1000)` (strict monotone increase) at **at
  least 2 of the 3 points**; AND
- at those points, `A(p, 1000) - A(p, 60)` exceeds **3 x SD(p, 60)**.

**Asymmetry rejected if:** `A` is flat or non-monotonic at 2 or more
points, or the increase clears the 3-sigma bar nowhere. Rejection settles
H4's asymmetry as absent for this build, at gap magnitudes spanning ~4x,
on a common design free of both defects above — the verdict the existing
data cannot support either way.

**Reported as context, not as part of the test:** absolute
`recovery_fraction` per point and window (expected to fall with window
length per Decision 2d's own two-timescale account, which this ladder
will show at a third magnitude rather than two), and per-point absolute
differences alongside the pooled numbers so the pooled-vs-clustered
distinction A3 raised is visible directly rather than argued abstractly.

**Secondary, on the same 90 runs.** Per-seed 2-state OU refit at all three
rungs, reporting **RMSE / peak_gap** (not absolute RMSE, per task 2 below)
at each (point, arm, window). Already known from the two magnitudes work
order 01 measured: this ratio fell (~0.0092 to ~0.0031, decreasing, not
growing) while absolute RMSE rose only 1.16x against a 3.5-4.4x gap
increase — the opposite of what a dropped term quadratic in the gap would
predict (~16x for a 4x gap). A third rung reads the trend across three
points instead of two, on the same common-design points the primary
statistic uses.

## Work order 02, task 1 — resolved: asymmetry CONFIRMED on a common design

**Superseded, not merely re-checked: work order 03's signed re-analysis
(below, "Work order 03, task 1") reverses this section's verdict to
REJECTED.** This section's `A(p,w,s) = |rec_eng - rec_comp| / mean(...)`
took an absolute value before averaging over seeds, which discards
direction and — on per-seed values scattered near zero — systematically
overstates magnitude (`E|X| > |E X|`). The signed statistic run on the
SAME 90 runs finds the two "confirming" points do not clear their own
threshold and do not even point the same direction. This section is kept
below for the record (it is what work order 02's pre-registration
committed to computing, and the numbers are correct given that
pre-registered statistic), not because its verdict stands.

All 90 runs complete (3 points x 2 arms x 3 windows x 5 seeds, fresh —
see the provenance table). Both analyses below follow the pre-
registration above exactly; neither was adjusted after seeing a number.

**Primary statistic, per point:**

| point | A(60) | A(300) | A(1000) | SD(60) | monotonic? | increase | 3xSD(60) | clears? |
|---|---|---|---|---|---|---|---|---|
| `consolidated_two_camp` LHS1 | 0.0589 | 0.1075 | 0.1927 | 0.0428 | yes | 0.1338 | 0.1284 | yes (+4.2%) |
| `cross_cut` LHS0 | 0.0571 | 0.0796 | 0.3010 | 0.0808 | yes | 0.2440 | 0.2424 | yes (+0.6%) |
| `cross_cut` LHS4 | 0.0293 | 0.0131 | 0.0743 | 0.0219 | **no** (dips at 300) | 0.0450 | 0.0656 | no |

**Verdict, applying the pre-registered rule exactly: ASYMMETRY CONFIRMED.**
2 of 3 points are monotonic (`consolidated_two_camp` LHS1, `cross_cut`
LHS0), and BOTH clear their own point's `3 x SD(60)` bar — the
pre-registration's own "confirmed" branch, not the "rejected" one.
**Stated precisely rather than rounded up: this is a narrow confirmation,
not an overwhelming one.** `cross_cut` LHS0 clears by 0.6% — a margin
inside ordinary seed-to-seed noise for a 5-seed SD estimate — and
`consolidated_two_camp` LHS1 clears by 4.2%, comfortable but not large.
`cross_cut` LHS4, the one point that does NOT confirm, is also the point
where `composition`'s own effect is largest relative to `engagement`'s
(the reverse of the other two) — noted as a fact about which point
behaves differently, not explained away with a new hypothesis this pass
has no data to test.

**Why this reverses Decision 2d's C2 null, not a contradiction of it:**
Decision 2d's own C2 threshold was itself methodologically sound in
INTENT (a pre-registered 3-sigma bar) but wrong in TWO other ways this
section's design fixes: harm and benefit were measured at DIFFERENT
points (confounding arm with design point — only 1 of 3 points
overlapped), and the 15-row-per-side statistics were pooled across 3
non-independent points rather than computed per point (A3's own
objection). This section's design uses the SAME 3 points for both arms
and computes `A` within each point, never pooled. Fixing those two
defects — not the units fix from task 2, which was about C1 — is what
surfaces the confirmation C2 missed. Decision 2d's own verdict stands as
recorded (a pre-registration is not retro-fitted); this section
supersedes it as the more soundly designed test of the same question.

**Context, not part of the test:** absolute `recovery_fraction` falls
with window length for both arms at all 3 points (e.g.
`consolidated_two_camp` LHS1: engagement 0.642->0.450->0.241, composition
0.638->0.498->0.293) — the same two-timescale mechanism Decision 2d
already established (more of the gap migrates into the slow `Bs` block
before withdrawal). The per-point absolute differences make the
pooled-vs-clustered distinction concrete: `cross_cut` LHS0's absolute
difference at window 1000 (0.0588) is larger than `consolidated_two_camp`
LHS1's (0.0516) even though the LATTER's relative asymmetry `A` is
smaller at every window — pooling the three points' absolute differences
together, as Decision 2d's C2 did, cannot see this because a large
absolute difference at a point with large `recovery_fraction` values
(diluting the relative signal) looks the same as a large absolute
difference at a point with small ones.

**Secondary: OU RMSE/peak_gap ratio across three points, not two — same
conclusion, more decisively.** Mean of the 6 per-point-arm ratios at each
window: **0.0177 (60) -> 0.0099 (300) -> 0.0050 (1000)** — a clean,
monotonic DECLINE, not the growth a dropped term quadratic in the gap
would produce. Fit quality stays high throughout (R² 0.82-0.94 at the
single lowest-signal seed, 0.95-0.999 typically). C1's own "residual
stays flat" conclusion is now checked at a third magnitude and holds
there too: the OU mechanism describes the larger gaps at least as well,
not worse, contrary to what a missing feedback term would predict.

**What this section does NOT do.** It does not re-litigate Decision 2d's
own two conclusions on their own terms (C1 stands, re-confirmed above;
C2 is superseded, not overturned by a flaw in the original reasoning —
the flaw was in the DESIGN feeding it). It does not explain WHY
`cross_cut` LHS4 fails to confirm while the other two do — a real
open question, not papered over with an unearned story. **It does not
reconcile this confirmed asymmetry with the OU DECAY mechanism's own
sign-symmetry (§5a, above), which this section leaves standing rather
than treating as contradicted:** `k`/`k_b` still do not know or care
about the sign of `Δ`, and nothing here re-measures that. `recovery_
fraction` depends on how `Δ_withdrawal` is SPLIT between the fast and
slow blocks at the moment of withdrawal, not only on the shared decay
constants — and `engagement`'s and `composition`'s own EXOGENOUS
accumulation processes (hostile contact vs. civil contact, entirely
different mechanisms upstream of withdrawal) could populate that split
differently without the shared post-withdrawal decay being asymmetric at
all. This is a plausible account, consistent with Decision 2d's own
`ΔBs`-share finding, not a demonstrated one — decomposing `ΔBs`/`Δ_with
drawal` per arm at all three rungs would test it directly and is not
done here.

**Demonstrated, not merely plausible: see "Work order 03, task 2,"
below.** The split ratio does differ systematically between arms,
tracks window length, and its sign matches each point's own recovery
asymmetry exactly (the arm with the larger slow-block share is the same
arm that point's own signed statistic calls stickier) — this also
supersedes this section's own earlier verdict, since the "asymmetry" it
CONFIRMED is itself superseded by work order 03's signed re-analysis to
REJECTED (with the rejection itself further characterized, not reversed,
by work order 04's sign-dependence finding).

## Work order 03, task 1 — resolved: the signed statistic reverses the confirmation to REJECTED

Re-analysis only, from the same `wave_c_gap_ladder.csv` (90 rows) work
order 02 collected — no new simulation. `A(p,w,s)` took `|rec_eng -
rec_comp|` before averaging over seeds; direction was discarded. The
signed statistic keeps the same seed-pairing (unchanged — it was the
right call) and drops only the absolute value:

    S(p, w, s) = (rec_eng(p,w,s) - rec_comp(p,w,s)) / mean(rec_eng(p,w,s), rec_comp(p,w,s))
    S(p, w)    = mean over s   [n=5]
    SD(p, w)   = sample SD over s, ddof=1

Positive `S` means `engagement` (harm) recovers MORE than `composition`
(benefit) — the direction OPPOSITE H4's claim that the hostile regime is
stickier. Negative `S` is H4's own predicted direction.

**1. `S(p,w)` and `SD(p,w)`:**

| point | S(60) | S(300) | S(1000) | SD(60) |
|---|---|---|---|---|
| `consolidated_two_camp` LHS1 | +0.0096 | -0.0930 | -0.1927 | 0.0778 |
| `cross_cut` LHS0 | +0.0556 | +0.0647 | +0.3010 | 0.0821 |
| `cross_cut` LHS4 | +0.0090 | -0.0080 | +0.0743 | 0.0381 |

**2. Sign consistency: not consistent, and the flip lands exactly on the
point work order 02 leaned on hardest.** Of the 45 (point, window, seed)
triples, 26 (57.8%) are positive (anti-H4 direction), 19 (42.2%) negative
(pro-H4). At the (point, window)-mean level:

- `cross_cut` LHS0 is positive at all 3 windows and grows MORE positive
  with exposure (+0.056 -> +0.065 -> +0.301) — anti-H4 throughout, and
  more strongly anti-H4 at the longest window. This matches Decision 2d's
  old pooled direction (harm 0.2339 > benefit 0.2121, also anti-H4) and
  simply extends it.
- `consolidated_two_camp` LHS1 starts positive (+0.0096, anti-H4, and
  indistinguishable from zero) at window 60, then FLIPS to negative at
  300 and 1000 (-0.093, -0.193) — into the pro-H4 direction, the OPPOSITE
  of Decision 2d's old pooled direction and the opposite of LHS0's own
  sign.
- `cross_cut` LHS4 oscillates (+, -, +) with no stable sign.

Two of the three points do not agree with each other on which arm is
stickier, let alone with a single directional claim. Work order 02's `|A|`
statistic could not see this because both a growing anti-H4 effect
(LHS0) and a late-emerging pro-H4 effect (LHS1) look identical once the
sign is discarded — `|A|` growing was reported as "the asymmetry," but
the two points that grow are not growing the SAME asymmetry.

**3. Monotonicity + threshold, re-run on `S`:**

| point | monotone? | \|S(1000)-S(60)\| | 3×SD(60) | SE of that bar (±35%) | clears? |
|---|---|---|---|---|---|
| `consolidated_two_camp` LHS1 | yes (monotone decrease) | 0.2024 | 0.2335 | ±0.0826 | no (-13.3%) |
| `cross_cut` LHS0 | yes (\|S\| increases, same sign) | 0.2454 | 0.2462 | ±0.0870 | no (-0.3%) |
| `cross_cut` LHS4 | no (sign oscillates) | 0.0654 | 0.1143 | ±0.0404 | no |

Same 2 of 3 points are monotonic as under `|A|` (LHS1, LHS0) — the
correction does not change WHICH points grow, only what growing means.
But under the signed statistic, NEITHER clears its own point's `3×SD(60)`
bar: LHS1 misses by 13.3%, LHS0 by 0.3% (a bare miss, the mirror image of
its bare 0.6% CLEAR under `|A|`). `cross_cut` LHS4 does not resolve to
monotone — the work order's own hypothesis that its dip under `|A|` was a
sign-crossing artifact does not hold; its sign genuinely oscillates
(+0.009, -0.008, +0.074), which is a different and messier pattern than a
single crossing. **The ladder does not become 3/3.**

**PRIMARY VERDICT ON THE SIGNED STATISTIC: REJECTED.** Fewer than 2 of the
3 monotonic points clear the magnitude bar (0 of 2) — the pre-
registration's own rejection branch, not a downgrade to "ambiguous."

**4. The verdict CHANGES.** Work order 02's task 1 reported CONFIRMED;
the signed re-analysis of the identical 90 runs reports REJECTED. The
work order's own prediction ("it probably does not [change]") does not
hold here — the direction check is what changes it, not just the
narrower magnitude test. Both facts matter and neither alone would have
been enough: the magnitude bar is missed by margins (13.3%, 0.3%) of the
same order as the bars' own ±35% estimation error, so a magnitude-only
re-run at n=5 would have been a coin flip either way; what removes the
ambiguity is that the two nominally-growing points do not even agree on
which arm is stickier, which no amount of additional precision on `A`
alone (an unsigned quantity) could ever have revealed.

**5. Threshold uncertainty, stated for the record.** `SD(p,60)` comes
from 5 seeds; its own standard error is `SD/sqrt(2x4)`, about 35% of
itself, so the `3xSD(60)` bar's standard error is about 35% of the bar.
Under the ORIGINAL `|A|` statistic, both reported "clears" were inside
that band: `cross_cut` LHS0 cleared by 0.6% of its bar (SE ±35%),
`consolidated_two_camp` LHS1 by 4.2% (SE ±35%) — both margins are
roughly an order of magnitude smaller than the bar's own uncertainty.
Under the signed statistic, both misses are of the same size and
therefore also inside that band. **The pre-registered rule's binary
output is not decisive in EITHER direction at n=5** — CONFIRMED and
REJECTED were both within one noisy threshold's own noise band of each
other. The signed re-analysis is reported as the standing verdict because
it corrects a real defect (direction-blindness) in the prior statistic,
not because its own margin is any more statistically decisive than the
one it replaces.

**6. Work order 04: the rejection is sign-dependence, not absence of
asymmetry — named as a result, not left implicit in a failed directional
test.** REJECTED is the correct pre-registered outcome (no single
direction holds across the space), but it is not the same finding as "no
asymmetry." Two of the three points show an asymmetry that grows
MONOTONICALLY with exposure duration, each internally consistent — they
simply grow in opposite directions, and the ordering lines up with the
one dial these three points span most:

| point | scenario | affective | ideological | structural | S(60) | S(300) | S(1000) | pattern |
|---|---|---|---|---|---|---|---|---|
| `consolidated_two_camp` LHS1 | consolidated | 0.946 | 0.885 | **0.732** | +0.010 | -0.093 | -0.193 | monotone, grows pro-H4 |
| `cross_cut` LHS4 | cross-cut | 0.544 | 0.864 | **0.298** | +0.009 | -0.008 | +0.074 | oscillates, no stable sign |
| `cross_cut` LHS0 | cross-cut | 0.671 | 0.587 | **0.013** | +0.056 | +0.065 | +0.301 | monotone, grows anti-H4 |

Sorted by the structural dial, the pattern reads: high structural
sorting (0.732) grows pro-H4 (harm becomes stickier — H4's own
prediction), near-zero structural sorting (0.013) grows anti-H4 (benefit
becomes stickier instead), and the middle value (0.298) is exactly where
a statistic that crosses zero somewhere in between would be expected to
show no stable sign. **This is the same shape as H1's own falsifier**:
a sign flip somewhere in the tribalization space is the substantive
finding, not an artifact to be explained away — H1 treats a flip in the
intervention's own effect as the interesting outcome; this is a flip in
a DIFFERENT quantity (which arm's post-withdrawal effect is stickier),
found the same way, in the same space.

**Numeric backing for "internally consistent," not just a read of the
table above (section 7's `mean(D)/SE`, carrying section 7's own
exploratory label):** all three points show a change in the signed
statistic across the ladder that is large relative to its own seed
noise — `mean(D)/SE` = -4.03 (`consolidated_two_camp` LHS1), +4.34
(`cross_cut` LHS0), +2.23 (`cross_cut` LHS4). Even LHS4, whose SIGN
oscillates, has a net change across the full ladder that is not
nothing — consistent with "no stable sign" (a mid-crossing point) rather
than "no effect." These are exploratory, post-hoc statistics (section 7)
and cannot revise REJECTED; they characterize what the rejection
consists of, which is what this section is for.

**The caveat this needs, stated plainly rather than left implicit:**
scenario and structural sorting are confounded in these three points.
`consolidated_two_camp` LHS1 is the only `consolidated_two_camp` point in
the ladder; both `cross_cut` points sit lower on the structural dial. So
"sign tracks structural sorting" and "sign tracks scenario" cannot be
told apart here, and the ordering above rests on 3 points, one of which
(LHS4) has no stable sign to order in the first place. **This is a
hypothesis the data suggests and this design cannot test, not a finding
in its own right** — recorded as a candidate account for the next
experiment to design around, not as part of what work order 03 verified.

**7. Work order 04, task 2 — EXPLORATORY, cannot change the verdict —
corrected in work order 05 (the first version compared `SD(D)` against
the wrong baseline):**
the pre-registration's `3xSD(p,60)` bar used the seed spread at the short
window alone; the quantity actually being tested is a difference between
two windows, so the paired per-seed difference is the more natural noise
scale:

    D(p, s) = S(p, 1000, s) - S(p, 60, s)   [seeds shared across windows]

| point | mean(D) | SD(D) | SE(mean D) | mean(D)/SE |
|---|---|---|---|---|
| `consolidated_two_camp` LHS1 | -0.2024 | 0.1123 | 0.0502 | -4.03 |
| `cross_cut` LHS0 | +0.2454 | 0.1264 | 0.0565 | +4.34 |
| `cross_cut` LHS4 | +0.0654 | 0.0657 | 0.0294 | +2.23 |

`mean(D)` reproduces `S(p,1000)-S(p,60)` from section 3 exactly, as it
must (a mean of paired differences equals the difference of means).

**Whether pairing helps is not answered by comparing `SD(D)` to either
endpoint's own `SD` — that was the wrong baseline, caught in work order
05.** A difference is noisier than its parts regardless of correlation;
`SD(D)` exceeding `SD(60)` or `SD(1000)` individually says nothing about
whether pairing helped. The right comparison is against the UNPAIRED
baseline `sqrt(Var(60) + Var(1000))` — the noise `D` would have if the
two windows' seeds were independent draws rather than the same 5 runs
measured twice:

| point | SD(60) | SD(1000) | unpaired baseline | SD(D) | pairing | implied r |
|---|---|---|---|---|---|---|
| `consolidated_two_camp` LHS1 | 0.0778 | 0.0582 | 0.0972 | 0.1123 | hurts | -0.35 |
| `cross_cut` LHS0 | 0.0821 | 0.1493 | 0.1704 | 0.1264 | helps | +0.53 |
| `cross_cut` LHS4 | 0.0381 | 0.0467 | 0.0603 | 0.0657 | hurts | -0.19 |

(`SD(1000)` for `cross_cut` LHS4 — not reported in the previous pass —
is 0.0467, from the same per-(point,window) aggregation as section 1's
table; it is now evaluable like the other two.)

Pairing HELPS at `cross_cut` LHS0 (`SD(D)`=0.126 against an unpaired
0.170, implied seed correlation +0.53 across windows) and HURTS at both
`consolidated_two_camp` LHS1 (implied r=-0.35) and `cross_cut` LHS4
(implied r=-0.19) — the two flaws in the earlier version of this
paragraph (comparing to the wrong baseline, and misreading LHS4's own
comparison as "below both" when 0.0657 is above both 0.0381 and 0.0467)
partly compounded: fixing the baseline is what makes LHS0's result read
as "pairing helps" rather than "in between," which the endpoint
comparison could not show either way.

**At n=5, none of this is distinguishable from zero, and that is the
actual scoping conclusion.** A correlation's standard error at n=5 is
roughly `1/sqrt(n-3)` = 1/sqrt(2) ~ 0.71 — both +0.53 and -0.35 (and
-0.19) sit well inside one SE of zero, and of each other. The honest
statement for a future design is **not** "pairing does not help" (the
first version's error) or "pairing helps, use it" (the naive read of
LHS0 alone) but: **the seed correlation across windows cannot be
estimated at this sample size — size a future design assuming
independent endpoints (the conservative, unpaired baseline), and treat
any pairing benefit that materializes as a bonus, not a planned
saving.** Per the work order's own labeling throughout: exploratory,
computed after the data was seen, and does not revise REJECTED (swapping
the noise scale after seeing two margins miss by 13.3%/0.3% under the
pre-registered scale is exactly the move pre-registration exists to
prevent) — its only role is scoping a successor design's precision.

**Recommended, not executed here — superseded by work order 04's own
correction:** adding seeds at the 60-tick rung would cut the bar's SE
from ~35% to ~19% at n=15, but that only sharpens a magnitude test at
these SAME 3 points. The open question §6 raises is different — whether
the sign tracks the structural dial — and more seeds at three confounded
points cannot answer that at any precision. The investment that could is
MORE POINTS: 6-8 points spanning the structural dial at roughly fixed
affective/ideological values, both scenarios represented at both ends
(breaking the scenario/structural confound), 2 windows (60 and 1000 —
the ladder already shows the middle rung adds little) rather than 3, 5
seeds, pre-registering a rank-correlation test on `sign(S(p,1000))`
against the structural coordinate rather than a per-point magnitude bar.
Either design is a new simulation increment needing its own
pre-registration before running, and neither is done as part of this
(analysis-only) work order. Whether it is worth running at all is a
separate, genuinely open call: H4 was one hypothesis among several in
Experiment 03's brief, it is now recorded as REJECTED on a sound design
with the sign-dependence noted as a candidate account for a successor
experiment, and the brief itself treats this withdrawal-protocol
statistic as a robustness check rather than a headline result — a weak
reason, on its own, to keep this experiment open further.

## Work order 03, task 2 — resolved: the ΔBs split account is DEMONSTRATED, not merely plausible

Work order 02's "What this section does NOT do" paragraph named a
candidate mechanism for how a confirmed asymmetry could coexist with the
OU decay's own sign-symmetry: `engagement` and `composition` are
different EXOGENOUS accumulation processes upstream of withdrawal, and
could split `Δ_withdrawal` between the fast and slow blocks differently,
without the shared post-withdrawal decay itself being asymmetric. That
paragraph explicitly flagged this as "plausible, consistent with Decision
2d's own `ΔBs`-share finding, not a demonstrated one." This task tests it
directly, from the same 90 runs, refitting the per-seed 2-state OU model
(same fit as `wave_c_gap_ladder_ou_fit.csv`) but keeping `dBs` and
`delta_peak` per seed instead of discarding them into a pooled RMSE.

**Split ratio `dBs/delta_peak`, mean over 5 seeds, by arm x point x window:**

| point | window | engagement | composition | diff (eng-comp) | SE(diff) | t |
|---|---|---|---|---|---|---|
| `consolidated_two_camp` LHS1 | 60 | 0.038 | 0.043 | -0.005 | 0.041 | -0.12 |
| `consolidated_two_camp` LHS1 | 300 | 0.332 | 0.228 | +0.103 | 0.026 | 4.00 |
| `consolidated_two_camp` LHS1 | 1000 | 0.641 | 0.572 | +0.069 | 0.009 | 8.14 |
| `cross_cut` LHS0 | 60 | 0.082 | 0.089 | -0.006 | 0.014 | -0.47 |
| `cross_cut` LHS0 | 300 | 0.338 | 0.383 | -0.045 | 0.024 | -1.89 |
| `cross_cut` LHS0 | 1000 | 0.657 | 0.748 | -0.091 | 0.026 | -3.53 |
| `cross_cut` LHS4 | 60 | 0.044 | 0.081 | -0.037 | 0.030 | -1.23 |
| `cross_cut` LHS4 | 300 | 0.359 | 0.357 | +0.002 | 0.009 | 0.24 |
| `cross_cut` LHS4 | 1000 | 0.659 | 0.682 | -0.023 | 0.003 | -6.65 |

(`diff`/`SE`/`t` computed on the seed-paired per-seed difference, same
pairing convention as `S(p,w,s)`.)

**The split DOES differ systematically between arms, tracks window
length, and — critically — its SIGN matches each point's own recovery
asymmetry exactly.** A higher `dBs/delta_peak` means more of that arm's
peak gap is trapped in the slow block (`k_b = k/10`), which mechanically
means LESS of it recovers in a fixed post-withdrawal window — i.e.
whichever arm has the larger split ratio at a point is the "stickier"
arm there, in the same sense `S(p,w)`'s sign measures:

- `consolidated_two_camp` LHS1: `engagement`'s split ratio EXCEEDS
  `composition`'s at 300 and 1000 (significant, t=4.00 and t=8.14) —
  `engagement` traps more of its gap in the slow block, so `engagement`
  is stickier. This is the PRO-H4 direction, and it is exactly the
  direction `S(p,w)` takes at this point (negative at 300, 1000 — see
  "Work order 03, task 1").
- `cross_cut` LHS0: `composition`'s split ratio EXCEEDS `engagement`'s at
  300 and 1000 (t=-1.89, -3.53), growing with window length —
  `composition` traps more of its gap, so `composition` is stickier.
  This is the ANTI-H4 direction, and again matches `S(p,w)`'s own sign at
  this point (positive throughout, growing).
- `cross_cut` LHS4: the diff is small at every window (|diff| <= 0.037)
  and does not grow monotonically (-0.037 -> +0.002 -> -0.023) — no
  consistent arm-level split asymmetry, matching this point's own
  unstable `S(p,w)` sign (section 6, "Work order 03, task 1").

**Verdict: DEMONSTRATED, per the work order's own stated criterion.**
The split differs between arms, the difference is significant and grows
with window length at the two points with a real `S` effect, and — the
strongest form of confirmation available — the ARM that gets the larger
slow-block share at each point is the same arm that point's own signed
recovery statistic calls "stickier." The account in work order 02's
"What this section does NOT do" paragraph is no longer a hedge; it is
the mechanism, and it now also explains WHY the sign of the asymmetry
flips between points: it is not that the shared OU decay becomes
asymmetric, but that `engagement`'s and `composition`'s own exogenous
accumulation processes populate the fast/slow split differently, and
which arm ends up more slow-block-heavy differs by design point — the
same points work order 04 found ordered by the structural dial. This
task does not itself explain WHY the structural dial predicts which arm
gets the larger slow-block share (that remains open, and is part of the
same confound work order 04 flagged: only 3 points, one
`consolidated_two_camp`, two `cross_cut`); it demonstrates the proximate
mechanism the earlier hedge could only propose.

**Provenance:** `results/experiment03/wave_c_gap_ladder_dbs_split.csv`
(90 rows, same runs and cache as `wave_c_gap_ladder.csv` and
`wave_c_gap_ladder_ou_fit.csv` — a per-seed refit keeping `dBs` and
`delta_peak` instead of only the pooled RMSE the earlier fit kept).

## Change spec V8.1-V8.4 — persisted `Bs`, `split_ratio`, the linked-metric registry, and phi's shape

Four items. The first three were fully specified; the fourth (`phi`) named
an open design fork that needed a decision before it could be written.

**V8.1 — `Bs` is now measured, not fit.** `DriftState.Bs`'s affect-block
columns persist on the traits frame as `animus_bs`/`identification_bs`
(`RUN_FORMAT` 6 -> 7), same shape as `x_i`, no new file. `Bs[t=0] ==
X_stored[t=0]` and `Bs[t+1] - Bs[t] == k_b*(X_stored[t+1] - Bs[t])` are
pinned directly on `DriftState` (`tests/test_drift.py`); persistence and
the used-space link conversion are pinned on an actual run
(`tests/test_posts_store.py`). A run without the affect block, or with
drift off, degrades correctly (no columns; falls back to `X_stored`
itself, which is what `Bs` conceptually equals when it never moves).

**V8.2 — `split_ratio` replaces the notebook-cell-46 least-squares fit.**
`_hysteresis_from_arrays` (`experiments/experiment03_bubble_intervention.py`)
now reads the MEASURED `Bs` gap at the peak tick instead of fitting the
2-state OU model's one free parameter against the post-withdrawal
trajectory. `recovery_fraction` stays as a derived convenience field,
`== b * (1 - split_ratio)` for `b` from `recovery_b(k, n_ticks)`, checked
on an actual 5-seed-averaged run (`tests/test_experiment03.py`): mean
`recovery_fraction` 0.174 against `b*(1-mean split_ratio)` 0.176 at
`n_users=800`, dial (1.0, 0.7, 0.9), `engagement`/`distance` — within
0.03, well inside the tolerance the conformance test uses. The identity
held: the 2-state OU model genuinely describes the post-withdrawal
dynamics at this design point, which is what cell 47 previously claimed
without being able to show.

**V8.3 — the linked-metric registry, seeded with all four known links.**
`registry.register(..., derives_from=...)` records the clean parent/child
case (`recovery_fraction` derives from `split_ratio`); `registry.
ALGEBRAIC_LINKS` records the three messier cross-metric identities
(`ingroup`/`outgroup` in `affect_delta`'s `"distance"` mode,
`toward_mean`/`toward_own_pole`, `algorithmic_share`/`rank_penalty`).
`registry.check_comparison(a, b)` raises on any of the four —chosen over
warning, per the spec's own reasoning that a warning here would be
ignored on the evidence of every other warning in this codebase.

**V8.4 — decision (a) adopted, after auditing the premise it rests on.**
The constraint: `phi(d) = d/(d+d0)` bounds the same/cross ratio below by
`d_same/d_cross` for ANY `d0` — the saturating SHAPE is the constraint,
not the calibration. The fork was between calibrating a logistic sigmoid
on the population's own pairwise-distance distribution (mechanism change,
re-runs everything after V7.3) and leaving `phi` alone and narrowing the
claim instead (costs nothing, leaves the bimodality gate as a hard
discontinuity). The open weakness named in the spec — whether
`wave_a_prime_recal`'s populations are bimodal enough at the design
points in use for a median-centred sigmoid to actually separate the
classes — was checkable, and was checked (`tests/test_change_spec_v8.py::
test_phi_sigmoid_separates_realized_camps_on_an_experiment03_scale_population`,
same audit shape as the `d_cross` pin test):

| ideological dial | bimodality (Sarle) | has camps | median-threshold separation accuracy | median vs. `d_cross` |
|---|---|---|---|---|
| 0.1 (below gate) | 0.34-0.36 | no | n/a — no classes to separate | n/a |
| 0.5 (the shared Wave A′ background) | 0.57-0.60 (barely above the 5/9 gate) | yes, weakly | 83-85% | median 4.5-7.0% below `d_cross` |
| 0.9 (fully bimodal) | 0.77-0.79 | yes | 93-98% | median within 0-14% of `d_cross`, usually <10% |

At design points where camps are clearly formed, a threshold at the
population's own median pairwise distance separates realized same-camp
from cross-camp pairs well (93-98%), and the median tracks `d_cross`
closely — the empirical premise decision (a) needs. Separation degrades
gracefully (not catastrophically) toward the gate, where camps are by
definition weakly formed and hard to separate under any statistic — a
limiting case, not a counter-argument. Below the gate the population is
genuinely unimodal, matching the design intent that `"distance"` mode
grades smoothly there rather than claiming out-group specificity that
does not exist yet.

**Implemented, not yet re-run.** `dynamics.affect_phi_shape: "saturating"
| "sigmoid"` (default `"saturating"`, byte-identical to pre-V8.4) and
`dynamics.affect_phi_width` (calibrated from the same pairwise-distance
sample's IQR as `affect_d0`, under `affect_d0_mode="calibrated"`) are
wired through `affect_delta` -> `apply_drift` -> `TickEngine`, mechanism-
level tested (`tests/test_change_spec_v8.py`). Actually switching
`affect_phi_shape` to `"sigmoid"` as the shipped default, and re-running
Wave A′/B/C and the hysteresis work under it, is a large compute
commitment (the recalibration re-run alone, a strictly smaller change,
took 2822s) explicitly left for a follow-up session with that budget
allocated, not executed here.

## Change spec V8.5 — `affect_phi_width`'s separability-scaled rule, and the honest ratio audit

Two changes to V8.4, both before the re-run budget was spent — neither
invalidates anything, since `affect_phi_shape` still defaults to
`"saturating"`.

**V8.5.1 — the IQR width rule replaced.** A fixed IQR divisor could not
serve both regimes: narrow enough (÷6) to beat the saturating shape's
0.435 same/cross floor when sorted, it gave a transition half-width of
±0.2 on a population with no camps at all below the gate — a step
function, reintroducing the discontinuity V7.3 removed. `dynamics/
tick.py` now calibrates `affect_phi_width` as `(mu_hi - mu_lo) / (6 *
eta)`, with `mu_lo`/`mu_hi`/`eta` from `metrics.polarization.otsu_
threshold_and_separability` on the same pairwise-distance sample
`affect_d0` is calibrated from — deterministic, no fitting, no camp
labels. `phi_width_from_separability` (`dynamics/drift.py`) reproduces
the change spec's own worked example exactly (`eta=0.90` → `w=0.55`,
`phi_same=0.063`, `phi_cross=0.937`, `ratio=0.067`, all at `d0=(mu_lo+
mu_hi)/2` — a pin, not just a shape check).

**V8.5.2 — the ratio re-audit, and the honest result.** Classification
accuracy (V8.4's own audit) fixes where `affect_d0` sits and says nothing
about the width; the number the change lives or dies on is the realized
same/cross `phi` ratio, measured directly on `wave_a_prime_recal`-scale
populations (`n_users=1000`, same ideological sweep as before, 5 seeds
each):

| ideological | bimodality | has camps | `eta` | realized ratio | classification accuracy |
|---|---|---|---|---|---|
| 0.1 (below gate) | 0.34-0.36 | no | 0.64-0.66 | 0.61-0.69 | n/a |
| 0.5 (shared background) | 0.57-0.60 | weakly | 0.69-0.71 | 0.32-0.34 | 0.83-0.85 |
| 0.7 | 0.69-0.71 | yes | 0.76-0.78 | 0.21-0.23 | 0.91-0.94 |
| 0.9 (sorted) | 0.78-0.79 | yes | 0.83-0.85 | 0.14-0.18 | 0.93-0.98 |

**Neither of V8.5.2's own gate criteria is met.** At ideological=0.9 the
ratio is ~0.16, not below the 0.10 target — a real improvement over
`"saturating"`'s 0.435 floor (roughly a 2.7x reduction), but short of
the bar. Below the gate, the ratio is ~0.65, closer to 1 (no manufactured
separation) than to 0, but not "near 1" as targeted either. A synthetic
population with less noise relative to signal (2 undifferentiated noise
axes alongside 1 camp-carrying axis, this project's own generator shape,
but with a larger separation-to-noise ratio) clears the 0.10 target
easily (`eta≈0.96`, ratio≈0.045 — `tests/test_change_spec_v8.py`),
confirming the shortfall is this population's own noise-floor geometry
at the scale in use, not a bug in the width formula: `otsu_threshold_
and_separability` reproduces the spec's own worked numbers exactly, and
`eta` at ideological=0.9 (0.83-0.85) sits meaningfully below the spec's
own illustrative `eta=0.90` row, which is most of the gap between the
achieved ~0.16 and the hoped-for ~0.067.

**Per V8.5.2's own gate ("do not start the Wave A′/B/C re-run until (1)
is below 0.10 in the sorted regime and (3) is near 1 below the gate"):
that re-run is NOT started.** `affect_phi_shape` stays at its
`"saturating"` default; nothing downstream is invalidated. Closing this
gap — a wider separation-to-noise ratio in the population generator, a
different width formula, or accepting a higher target ratio — is future
work, not attempted here to avoid curve-fitting a threshold this
session did not set.

**Provenance:** `tests/test_change_spec_v8.py::
test_phi_v8_5_1_realized_ratio_on_wave_a_prime_recal_scale_populations`
pins the measured range so a future change to the width formula, the
population generator, or the scale in use shows up as a test change, not
a silent drift in an unmeasured number.
