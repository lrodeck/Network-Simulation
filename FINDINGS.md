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
