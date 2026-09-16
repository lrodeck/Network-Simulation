# Pre-submission checklist — run this against FINDINGS.md before handing it back

Derived from two rounds of review (Review 01, Review 02) plus
experiment-03-bubble-intervention.md §7. Fifteen items in four groups (14
original + C5, added after the first run of this checklist exposed a gap in
its own coverage). Every item recurred at least twice across the two
reviews, which is why it is here and not in a general style guide.

**How to run it.** After writing a new findings section and before submitting,
walk every item against that section specifically. For each: either the check
passes, or the section says in its own words that it does not and what that
costs. A stated limitation passes. A silent one does not. Report the result as a
short table at the end of the submission — item, pass/stated-limitation, one
line — so the reader knows the screen ran, not just that the work happened.

Items marked **[blocking]** mean: do not submit the section claiming a result
until the item passes or the claim is downgraded to match what the item found.

---

## How to score an item

*Added after the first run of this checklist scored three items **pass**
that were not passes, all three carried over from a prior review rather
than newly found.*

- **pass** means the check ran against this section, this pass, and found
  nothing. It does not mean the issue was raised somewhere else, flagged in an
  earlier section, or known. A known problem that has not been measured is a
  fail.
- **fixed** means the check ran, found something, and the section was changed.
  Say what changed.
- **stated limitation** means the check ran, found something, and the section
  now says so and what it costs. State it at its real size — if the limitation
  sits on the section's headline claim rather than a secondary one, say that.
- **fail** means the check did not run, or ran and the section was not
  changed.

An item carried over from a previous review cannot be scored **pass** on
reasoning alone. It needs a measurement made this pass, or it is a fail with a
named next step. If an item was scored pass and the underlying claim is
unchanged in the text, that is a fail regardless of the reasoning attached.

Scoring an item against a *neighbouring* check is the specific way this went
wrong the first time: measuring one consequence of a miscalibrated constant
does not discharge an item about the constant.

---

## Group A — Declared rules that stop getting applied

The brief declares decision rules and the findings stop applying them. This
produced Review 01 items 2, 3 and 4, and Review 02 item 4.

**A1. Every sign claim is paired with a magnitude and a SESOI verdict.**
**[blocking]**
§7's rule: smallest effect size of interest is 10% of the `none` arm's own drift
over the same window, at the same design point. For each headline claim, report
the effect, the `none` arm's own drift, the threshold, the ratio, and a
clears/does-not-clear verdict. A consistency count (40/40, 113/120) on its own is
not a result — sign consistency is cheap when an effect is tiny but systematic.
*Failure signature:* a paragraph that reports only how many cells share a sign.

**A2. Consistency counts exclude structurally-degenerate cells.**
A cell where one arm is an exact zero (camp gate, missing feature, NaN-gated
outcome) is not evidence about the comparison being counted. Count over
measurable cells only; list the degenerate ones separately with the reason.
*Failure signature:* an n/n count that includes a cell where one side is
exactly 0.0.

**A3. Seeds within a design point are not independent draws.**
§7: cluster by seed — runs sharing a seed share a population and a graph. Any
claim resting on cell counts should say what varies across the counted units. 5
seeds at one design point is one population measured five times.
*Failure signature:* "40/40 seed-cells" quoted as if it were 40 observations.

**A4. Per-contact normalization is applied and its denominator is characterized.**
§7 requires per-contact normalization on every animus claim. Beyond applying it:
report what fraction of events the denominator actually selects. A restricted
denominator that selects ~everything is not a robustness check — it reproduces
the unrestricted one by construction.
*Failure signature:* "normalization-invariant across all three measures,"
without the selection fractions.

**A5. Each comparison is against its own matched reference, not a pooled one.**
§7: no comparison against a pooled threshold; each design point is its own
reference. Includes matched-null discipline (`quality_attention_lift`'s assertion
is the house precedent) and arm-minus-`none` differencing at the same point.
*Failure signature:* a median or threshold computed across design points and then
applied within them without saying so.

---

## Group B — Constants calibrated in one regime, reused in another

This is the project's single most expensive recurring failure: `attention_budget`
at 15/30/60 (a dead-zone sweep), `homophily_beta` (inert), `mirror_p` vs
`sbm_mirror_p` (never read, docstring wrong), and `affect_d0 = 1.0` at D=3. Each
was invisible until someone measured the constant instead of trusting it.

**B1. Every new or reused scalar constant is measured in the regime it is
actually running in.** **[blocking]**
For each constant a new section depends on: what regime was it calibrated for,
what regime is it running in now, and what does it measure in the current one?
Distances, thresholds, rates and scales all change meaning with dimension,
population size, tick count and marginal shape.
*Failure signature:* "calibrated per the spec for a one-axis standard-normal
population" appearing in a section that runs at D=3.

> **Flagged is not measured.** *(added after the first run)* An acknowledged
> bad constant is the case this item exists for, not an exemption from it. If
> a previous section recorded that a constant is miscalibrated for the
> current regime, every later section whose results depend on it is under
> this item until the constant is measured in that regime.
>
> **"No new constant" is not the test.** The scope is every constant the
> current results depend on, whether introduced this pass or inherited. A
> default that changed underneath the results (a mechanism switched on by
> default in an earlier wave) is inherited, not exempt.
>
> **Measuring one consequence does not discharge the item.** A constant
> feeding two consumers needs both measured — that is B3, and B1 is not
> satisfied by partial coverage of it.

**B2. A threshold or midpoint constant is reported as a realized distribution,
not as its nominal value.**
If a constant splits events into classes (`d_cross`, `agree_delta`, a gate
threshold), report the realized split: what fraction lands each side, and the
contrast between the classes on the quantity the constant is supposed to
discriminate. A nominal midpoint says nothing about where the population sits
relative to it.
*Failure signature:* a threshold quoted by value with no distribution alongside.

**B3. A constant shared by two mechanisms is checked in both.**
Tying `d_cross` to `affect_d0` so they cannot drift apart is correct and keeps
them consistent — but it also means one bad calibration lands in two places.
When a constant is shared, check its effect in each consumer separately.
*Failure signature:* "so the two thresholds cannot drift apart," with only one of
the two measured.

**B4. Every consumer of a gated or conditional feature is enumerated before
claiming a gate was removed.**
V7.3's warrant — "the affect op is the last consumer of the camp label" — was
wrong, and Wave A′ found the second consumer (`compute_features`'
`CONDITIONAL_FEATURES`) only by accident of an exact-zero column. Before writing
"this gate is now removed," grep for every reader of the gating condition and
list them.
*Failure signature:* a claim that a mechanism now works below a threshold,
without an enumeration of what else reads that threshold.

---

## Group C — Conclusions from models that omit the mechanism in question

Review 02 item 3 (H4), and the brief's own §6 tautology warnings.

**C1. A derivation that predicts a hypothesis's outcome must contain the
mechanism the hypothesis is about.** **[blocking]**
Before concluding "H-x is not supported," name the mechanism that could produce
the effect, and confirm the model or argument used to reach the conclusion
contains it. If the derivation drops it — a linearization, a cancellation that
assumes state-independence, a fixed coefficient — the conclusion is circular.
*Failure signature:* "the recursion is linear and sign-symmetric, so equal
recovery is predicted," under a config where the gain term reads state.

> **Scored on content, not production.** *(added after work-order-01)* C1
> asks whether the derivation CONTAINS the mechanism, not when or in which
> commit it was produced. A derivation computed earlier in the same
> investigation, reused unchanged, is not "reasoning about a flag" the way
> B1's amendment warns against — it is still a measurement, just an
> earlier one. Score it by re-reading what the derivation actually
> contains; do not downgrade a genuine derivation only because it predates
> the section citing it, and do not upgrade a stale flag only because it
> is recent.

**C2. "Not supported" and "not testable here" are distinguished explicitly, per
hypothesis.**
The `delta_k` / `emergent_camps` paragraph is the house standard: it states that
no generator produces more than two modes, so the zero is structural and
uninformative rather than a null. Apply the same standard everywhere. A result
that could not have come out otherwise is not a result.
*Failure signature:* a falsifier reported as not firing, in a build where it
could not have fired.

**C3. An arm's own construction is separated from the effect being attributed to
it.** The brief's §3 warning: the engagement arm will and should show backfire by
construction. For each arm, state what it does by construction and what remains
after that is accounted for. The composition-vs-engagement contrast is the house
example of doing this correctly — one lever differs, so the difference is the
result.
*Failure signature:* an absolute arm-vs-`none` direction reported as a finding
where the arm manipulates the measured quantity directly.

**C4. An alternative explanation is stated and separated before a methodological
finding is named.** When a section produces a nameable finding ("camp-aware
targeting cannot act on an unsorted population"), write down the dullest
competing explanation and say what distinguishes them in the data. Name it only
after the competitor is ruled out.
*Failure signature:* a finding given a name and bolded, with no competitor
considered.

**C5. A null on a summary statistic is reported with a sensitivity check.**
*(added after the first run)*
Before reading a flat statistic as a real null, show the statistic moves when
the underlying quantity moves. Report it alongside a quantity known to vary in
this data, across a range where that quantity varies substantially. Also state
the statistic's own convention — post-vs-pre within an arm, or arm-vs-`none` —
since an arm-vs-`none` ratio differences out exactly the shared movement a
null would need to rule out.
*Failure signature:* a ratio flat to within a few percent everywhere, read as
the two outcomes being orthogonal, in a document that elsewhere reports the
underlying quantity varying by an order of magnitude or more.
*Current instance:* `diversity_floor_break_even` at 0.997–1.024 across all 90
hostility-reducing Wave B rows, against `ideo_level_toward_own_pole` spanning
−0.0004 to +0.0262 with a ~65× scaling on the affective dial.

---

## Group D — Effective degrees of freedom below nominal

Review 01 item 4, Review 02 item 2.

**D1. Any fit reports its effective design points, not its row count.**
**[blocking]**
Points clustered in tight neighborhoods contribute roughly one effective point
per cluster at the between-cluster level. Report parameters against effective
points. If parameters ≥ effective points, the fit is under-determined and its R²
is not evidence — say so instead of reporting coefficients.
*Failure signature:* a high R² from a design with fewer distinct neighborhoods
than parameters.

**D2. Fits are validated out of sample or with the clustering absorbed.**
Refit with cluster/scenario fixed effects so coefficients come from within-
cluster leverage, or hold out a cluster and predict it. Report that number
alongside the raw R².
*Failure signature:* a single R² carrying an interpretive claim.

**D3. Algebraically dependent quantities are counted once.**
Mirror-image components (`toward_other_camp` / `toward_mean` /
`toward_own_pole` at a two-camp near-linear structure) are one signal. Pick the
primary, note the others as implied.
*Failure signature:* a conjunction of two components presented as two
confirmations.

**D4. Good and bad fits are held to the same standard.**
If a low R² is reported as an honest negative and its coefficients withheld, a
high R² from the same design cannot be reported as informative without the
Group D checks. Same design, same scrutiny, both directions.
*Failure signature:* asymmetric treatment of two fits from one design.

---

## Standing requirements, not per-section checks

- **Dataset provenance.** Every number traces to a named dataset in a table that
  lists substrate, mechanism, background, and which dataset is canonical. When a
  re-run is not byte-identical to its predecessor, the table says why.
- **Stopping rules before the next wave, not after.** For any hypothesis
  repeatedly deferred to a larger design, write the coverage that would settle it
  before running the next wave. A falsifier deferred three times is not being
  applied.
- **Selection on noisy estimates is labelled.** Points chosen as extremes of a
  low-seed estimate are "the largest n-seed estimates," not "the largest
  effects."
- **Derived artifacts move in the same pass as the findings, not after.**
  *(added after work-order-01)* A notebook, a summary table, a verdict row —
  anything that restates a finding in a second place — is corrected in the
  SAME commit that corrects the finding itself. A prose fix with the
  notebook cell or table row still reading the old claim is a fail, found
  on whatever later pass actually reads that second place (this is exactly
  how the B1 units bug and the C5 domain bug each surfaced twice: once in
  prose, once in a derived artifact that had not been told about the fix).
- **A pre-registered threshold is specified in units invariant to the
  manipulation being tested.** *(added after work-order-02)* If the
  experiment changes a statistic's scale, a threshold fixed in that
  statistic's absolute units is not the test that was registered. Before
  registering, ask what the manipulation does to the statistic's mean and
  spread; if either moves, register a ratio, a fraction-of-mean, or a
  within-run standardized quantity instead.
  *Failure signature:* a threshold derived from a baseline run's SD, applied
  to a run whose SD the manipulation was expected to change.
  *Current instance:* C2's 0.081 bar, set at 3x the short-gap pooled SD,
  applied to larger-gap data whose SD had fallen by ~2.5x, where it
  functioned as an 8-sigma bar.
- **A statistic that tests a directional claim is not passed through
  `|·|` (or any other convexity) before averaging over noisy units.**
  *(added after work-order-03)* `E|X| > |E X|` — taking an absolute value
  first turns pure per-seed noise around a true value of ~0 into apparent
  magnitude, and can manufacture a spurious "monotonic increase" purely
  from the true effect growing (so noise's relative share of `|X|`
  shrinks), with no way to tell that story apart from a genuine growing
  asymmetry until the sign is checked. Average the signed statistic first;
  apply `|·|` afterward only if magnitude, not direction, is what the
  hypothesis actually needs — and if the hypothesis IS directional, never
  apply it at all.
  *Failure signature:* two design points both showing `|A|` "growing with
  exposure," reported together as one confirmed asymmetry, when their
  signed values are growing in OPPOSITE directions.
  *Current instance:* work order 02's H4 gap-ladder statistic
  (`A = |rec_eng - rec_comp| / mean(...)`) reported CONFIRMED at 2 of 3
  points; the signed version of the same 90 runs found neither point
  clears its own threshold, and the two do not agree on which arm is
  stickier — work order 03, task 1.
- **The existing house virtues stay.** Record the gap before closing it; test
  first; default new mechanisms off; state non-comparability at the point of
  citation. Those are working and are why the close-out rate on Review 01 was
  6/6.
