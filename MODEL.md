# The model, explained

What this simulation is, what every variable means, and the mathematics behind
each mechanism.

**How to read this.** Every section starts in plain language and needs no
mathematics. The formal statement is folded into a `▸ The math` block you can
click open or ignore entirely — nothing later depends on having read one. Terms
are defined where they first appear.

**Contents**

1. [What the model is](#1-what-the-model-is)
2. [The people](#2-the-people)
3. [The network](#3-the-network)
4. [Time and who speaks](#4-time-and-who-speaks)
5. [What a post is](#5-what-a-post-is)
6. [Replies and threads](#6-replies-and-threads)
7. [The feed](#7-the-feed)
8. [The engagement kernel](#8-the-engagement-kernel-the-theory-of-why-people-react)
9. [Cascades](#9-cascades)
10. [The public mood](#10-the-public-mood)
11. [Drift: how people change](#11-drift-how-people-change)
12. [One tick, start to finish](#12-one-tick-start-to-finish)
13. [Measuring it](#13-measuring-it)
14. [Every knob, in one table](#14-every-knob-in-one-table)
15. [What this model cannot tell you](#15-what-this-model-cannot-tell-you)

---

## 1. What the model is

A simulated social-media platform. A population of users, each with a
personality and a set of political positions, follows one another, posts,
sees a feed, and reacts. Run it for a few hundred rounds and you can ask:
**does this platform put people in front of those they disagree with, and who
gets heard?**

Then change one design choice — how the feed is sorted, how much random content
is injected, how far people scroll — hold everything else fixed, and ask again.
That comparison is the point. It is a *normative* study: not "what does social
media do", but "which platform design choices support democratic discourse".

Three things it deliberately is **not**:

- **Not agentic.** No language model decides anything. Every choice a simulated
  person makes is drawn from a probability distribution with named parameters.
  That is what makes it fast (whole populations at once, not one agent at a
  time), reproducible, and analysable.
- **Not a predictor.** It will not tell you what Twitter will do next month. It
  is a laboratory for mechanisms: it tells you what *follows* from a set of
  assumptions you can read and argue with.
- **Not language-driven.** Posts are numbers — a topic, a position, a level of
  arousal. Turning those into readable text is a separate, optional pass that
  runs *after* a simulation, never during it.

### The two-layer design

| | Simulation layer | Language layer |
|---|---|---|
| **What it handles** | who posts, who sees what, who reacts | turning a post's numbers into words |
| **When it runs** | every tick | offline, after the fact, optional |
| **Uses an LLM** | never | yes |
| **Reproducible** | exactly, from a seed | not exactly |

Keeping these apart is a hard rule. It is why a 500-tick run of 10,000 users
finishes in minutes and why the same seed gives the same answer every time.

---

## 2. The people

Every user is a **row of numbers** — roughly 35 of them, grouped into six
blocks. Nothing else about a person exists in the model.

| Block | Columns | What it captures |
|---|---|---|
| **personality** | openness, conscientiousness, extraversion, agreeableness, neuroticism | the Big Five; drives *how* someone writes |
| **expression** | verbosity, formality, irony, humor, profanity, emoji | writing style |
| **topic_affinity** | one per topic (8 by default) | how much each subject interests them |
| **stance** | one per ideological axis (3 by default) | where they stand politically |
| **behavior** | activity, reply_prop, repost_prop, contrarianism, credulity, prominence | how much and how they participate |
| **meta** | plasticity, conviction, circadian_phase | how changeable they are, and when they are awake |

Two of these carry most of the model's weight:

- **activity** — how often someone posts. It is drawn *lognormal*, meaning a
  few people post enormously more than everyone else. This single choice
  produces most of what naively looks like emergent structure, which is why
  every result is compared against a null model that keeps it (see
  [§13](#13-measuring-it)).
- **prominence** — how likely others are to follow you. Drawn *Pareto*: a
  small number of accounts are far more followable than the rest.

### Stance: the political space

The most important design choice. A user's political position is not one number
but **D numbers** — a point in a *D*-dimensional space. By default D = 3, and
a loaded scenario names each axis and each of its two poles:

| Axis | Negative pole | Positive pole |
|---|---|---|
| provision | market | state |
| openness | closed | open |
| institutional trust | distrust | trust |

So a user at `(−1.2, +0.4, −0.8)` leans market, slightly open, distrusting.
"Agreement" between two people is how *close together* they are in this space.

> **Why D matters more than it looks.** At D = 1 there is no orientation for
> homophily to be homophilous *in* — everyone is on a single left–right line and
> the geometry of the model is genuinely different, not just smaller. Loading a
> scenario overrides the configured D, and the library refuses to do that
> silently for exactly this reason.

<details>
<summary><b>▸ The math: how the population is drawn</b></summary>

Each column *j* has a **target marginal distribution** — the shape that column
should have across the population (lognormal for activity, Pareto for
prominence, Beta(2,2) for most bounded traits, Normal for personality and
stance).

Drawing them independently would make everyone's traits uncorrelated, which is
wrong: agreeable people really are less provocative. So the population is drawn
with a **Gaussian copula**:

1. Draw `Z ~ N(0, Σ)` — a multivariate normal with the desired correlation
   structure Σ.
2. Map each column through its own normal CDF to a uniform: `U_j = Φ(Z_j)`.
3. Map that uniform through the target marginal's inverse CDF:
   `X_j = F_j⁻¹(U_j)`.

Step 3 gives each column exactly the marginal you asked for; step 1 gives the
joint structure. Σ is assembled from `correlation_pairs` and projected to the
nearest positive-semidefinite matrix (Higham's algorithm) if the requested
correlations are jointly impossible.

On top of this sits an **archetype mixture** — named groups (lurker, poster,
firebrand, institution, newcomer) with per-trait offsets, so the population is
drawn from a mixture rather than a single blob.

> ⚠️ **The two mechanisms compose additively and neither knows about the
> other.** An archetype that shifts two traits together already correlates
> them. Asking for `activity × reply_prop = 0.30` with the shipped archetypes on
> yields **0.49**. Control a pair with one mechanism or the other, not both.
> `sample_population` warns when you do.

**Two coordinate systems.** Each trait is stored twice:

- `X_used` — the constrained value the model reads (a probability in [0,1], a
  positive rate).
- `X_stored` — an unconstrained value, related by a **link function**
  (`logit` for [0,1], `log` for positives, `identity` for reals).

Drift ([§11](#11-drift-how-people-change)) operates on `X_stored` and is plain
addition there, so it can never push a probability past 1 or a rate below 0.

</details>

---

## 3. The network

Who follows whom. Two forces decide it, and they are the two forces that
actually structure real follow graphs:

- **Homophily** — you are more likely to follow someone close to you in
  stance *and* topic interest.
- **Prominence** — you are more likely to follow someone many others follow.

Plus a third that matters more than its size suggests: a fraction of ties
(10% by default) are drawn **completely at random**. Without these, the network
is a set of sealed neighbourhoods and nothing ever spreads beyond one. They are
the shortcuts.

The graph is directed: `u → v` means *u follows v*, so v's posts reach u.

<details>
<summary><b>▸ The math: the latent-space generator</b></summary>

Place every user at the point given by their stance and topic-affinity columns
stacked together. Distance is ordinary Euclidean distance in that space:

```
d(u, v) = ‖ [s_u ; a_u] − [s_v ; a_v] ‖₂
```

The probability of a follow edge is a logistic function of distance and the
target's prominence:

```
logit P(u → v) = α − β · d(u, v) + γ · log(1 + prominence_v)
```

- **β** (`homophily_beta`, 0.35) — how strongly closeness matters.
- **γ** (`prominence_gamma`, 0.6) — how strongly popularity matters.
- **α** — not a free parameter. It is solved for by bisection so the finished
  graph hits the target `mean_degree` (40 by default).

Candidates come from a *k*-nearest-neighbour search (`knn_k = 60`) rather than
all N² pairs, which is the only tractable option past ~20k users. Long ties are
then added uniformly at random, and α aims at `mean_degree / (1 + long_tie_fraction)`
so the total lands on target rather than 10% over it.

> ⚠️ **`knn_k` must exceed `mean_degree`.** If the candidate pool is no larger
> than the degree drawn from it, every candidate is taken and β and γ have
> nothing left to select among — the generator silently degrades to plain kNN.
> Measured at `knn_k = 40, mean_degree = 40`, sweeping β from 0.35 to 1.5
> changed clustering by *exactly nothing*. The config now raises rather than
> letting this happen.

**A known ceiling, recorded rather than hidden.** `prominence` enters as
Pareto(2.30) with a max/mean of 303×, but realised in-degree comes out far
flatter (tail exponent ≈ 7.9). Within the kNN pool a user can only be followed
by the ~`knn_k` users whose neighbourhood contains them; the γ term reorders
those candidates but cannot lift anyone out of that geometric ceiling. Since
engagement per post cannot be more skewed than the audience sizes it is drawn
over, this is why the attention-Gini target is missed. See
[`FINDINGS.md`](FINDINGS.md).

**Alternative generators** (`cfg.graph.generator`): `latent_pa` (long ties drawn
proportional to prominence, lifting the tail), `sbm` (stochastic block model),
`configuration` (degree-preserving null), `barabasi` (pure preferential
attachment). Each is a useful null for isolating what homophily itself
contributes.

</details>

---

## 4. Time and who speaks

Time advances in **ticks**, one hour each (24 per day). On every tick each user
may post, with a probability set by three multiplied factors:

1. **their activity** — the lognormal trait; some people are simply louder;
2. **the hour** — a two-peak daily rhythm, busy around 9am and 7pm, and each
   user has their own phase offset, so the population is not synchronised;
3. **fatigue** — posting a burst suppresses your rate next tick, and it relaxes
   back over time.

<details>
<summary><b>▸ The math: posting rate</b></summary>

```
λ_u(t)      = activity_u · posts_per_tick_rate · circ(t − φ_u) · fatigue_u(t)
n_posts_u(t) ~ Poisson(λ_u(t))
```

`circ` is a sum of two von Mises bumps over the day (peaks at 0.38 and 0.79 of
the day, concentration κ = 6), **mean-normalised to 1** so it redistributes
activity across the day without changing anyone's long-run total.

`posts_per_tick_rate` (0.02) is the rate at `activity = 1`. It is a separate
scalar from the activity trait for a reason: without it the raw trait was the
rate, giving ~2 posts/user/tick instead of ~0.04, and averaging many Poisson
draws per user per tick pulls everyone toward the mean — which destroys exactly
the heterogeneity the lognormal trait exists to create.

Fatigue: `f ← decay·f + (1−decay)`, then `f ← f / (1 + n_posts)`.

</details>

---

## 5. What a post is

A post is also just numbers:

| Field | Meaning |
|---|---|
| `topic` | which of the K subjects (an integer) |
| `stance` | a point in the same D-dimensional political space as users |
| `arousal` | how emotionally charged |
| `valence` | positive or negative in tone |
| `provocativeness` | how much it invites a fight |
| `novelty` | how new the claim is |
| `specificity` | how concrete versus vague |
| `quality` | epistemic merit |
| `length` | how long |
| metadata | `id`, `t`, `author`, `parent`, `root`, `depth`, `kind`, `engagement_count` |

**Choosing a topic** blends the author's own interests with what everyone is
currently talking about. **Choosing a stance** blends the author's own position
with the position currently dominant on that topic — weighted by their
`conviction`. Someone with high conviction posts their own view; someone with
low conviction echoes the room.

**The style dimensions** come from a hand-written table of individually
arguable claims: high neuroticism raises arousal; low agreeableness raises
provocativeness; high conscientiousness raises quality and specificity.

<details>
<summary><b>▸ The math: post generation</b></summary>

**Topic** — a softmax over the author's affinities, tilted by the public agenda
`s(t)`:

```
P(topic = k) ∝ exp( a_u[k] + η · s(t)[k] )
```

`η` is `trend_eta` (0.3): how susceptible people are to what is trending.

**Stance** — a convex combination of the author's own position and the topic's
currently dominant position `σ(t)[k]`, with noise:

```
stance_p = conviction_u · s_u + (1 − conviction_u) · σ(t)[topic_p] + ε
```

**Style dimensions** — a linear map from author traits, the public mood and the
topic:

```
d_p = A · x_u + B · s(t) + C · onehot(topic_p) + ε_d
```

`A` is the single most important authored object in the system. It is a sparse
named-entry table, never a dense matrix, so every claim it makes is legible and
individually editable:

| post dim | trait | weight |
|---|---|---|
| arousal | neuroticism | +0.6 |
| arousal | extraversion | +0.3 |
| valence | neuroticism | −0.4 |
| provocativeness | agreeableness | −0.6 |
| provocativeness | contrarianism | +0.7 |
| novelty | openness | +0.5 |
| quality | conscientiousness | +0.4 |
| quality | credulity | −0.3 |
| length | verbosity | +0.8 |

(Abridged; unlisted pairs are zero.) Post dims are stored unconstrained with a
link per dim, the same discipline as user traits.

> ⚠️ **`quality` is generated from author traits.** So `corr(quality,
> engagement)` partly measures the data-generating process rather than the
> platform. It is interpretable **only** as a difference from the matched null.

</details>

---

## 6. Replies and threads

Replies are **not** produced by people browsing their feed. They come from a
separate process, because real conversations are bursty: a post gets its
comments in a clump, not spread evenly, and each reply makes the next one more
likely.

This is a **self-exciting process** (a Hawkes process). Each open thread carries
a running "heat" that jumps when a reply arrives and decays exponentially
otherwise.

Two dials govern it, and they interact in a way worth understanding:

- **`hawkes_ratio`** (0.6) — how much each reply excites the next *within* a
  thread. Must stay below 1 or threads never stop.
- **`hawkes_mu_inherit`** (1.0) — how much heat a reply's *own* new thread
  inherits from the thread it landed in. At 0 a reply inside a raging argument
  is as cold as a fresh post, so depth cannot compound and threads stay flat.

<details>
<summary><b>▸ The math: the Hawkes intensity</b></summary>

```
λ_p(t) = μ_p + Σ_{t_i < t} α · exp(−β · (t − t_i))
```

with `α = hawkes_ratio · β`, so the branching ratio `α/β = hawkes_ratio`.
Implemented by the standard exponential-kernel recursion — the sum collapses to
one decaying state variable per thread, so cost per tick is O(open threads),
not O(events).

**A stability subtlety.** `hawkes_ratio < 1` bounds excitation *within* a
thread. `hawkes_mu_inherit` adds a second channel *across* generations, so
`α/β < 1` alone no longer guarantees stability. Measured at
`hawkes_ratio = 0.6`: inherit 0.16 gives 2.4 replies per post; **inherit 0.20
gives 473**. The transition is sharp, so the tick warns rather than letting a
run silently saturate.

Inherited heat is seeded into `excitation` (which decays), never into `μ`
(which does not) — otherwise a thread would carry a permanent inherited
baseline.

**Why the default is not depth-optimal.** Deep threads dilute the thing the
experiments measure. Hawkes replies are not kernel-driven — a reply carries the
replier's own stance — so the more of the corpus is replies, the less of what a
user consumes was selected by the engagement kernel, and the null comparison
loses statistical power. Measured, the homophily agreement effect against its
matched null:

| `hawkes_mu_inherit` | mean depth | effect *t* |
|---|---|---|
| 0.6 | 1.16 | +3.77 |
| **1.0 (default)** | **~1.2** | **+2.46** |
| 1.8 | 1.43 | +1.93 |
| 2.65 | 2.30 | +0.74 (noise) |

Depth 1.5–3 is reachable and stable at inherit ≥ 2.2. It is an **experimental
condition to select deliberately**, not a default, because setting it costs the
null comparison its resolution.

</details>

---

## 7. The feed

Three steps between a post existing and someone seeing it.

**Step 1 — candidates.** A post goes to its author's followers, plus
`inject_k` randomly chosen non-followers ("recommended for you").

> This is why the "algorithmic share" of exposure is undefined at
> `inject_k = 0`: **injection is the only source of non-follower candidates**,
> so no ranker setting can create one. That is the feed's construction, not a
> measurement failure.

**Step 2 — ranking.** Each candidate gets a score; higher sorts first. The
ranker *is* the platform's central design choice:

| Ranker | Sorts by | In one line |
|---|---|---|
| `chronological` | post time | the reverse-chronological feed |
| `random` | noise | a control |
| `popularity` | engagement so far | pure bandwagon |
| `affinity` | topic interest + agreement | filter-bubble maximal |
| `engagement_optimized` | affinity + arousal + social proof | a stand-in for a trained propensity model |

**Step 3 — attention.** Nobody sees everything. Two caps apply: a personal
budget of items per tick, and a decay in how likely you are to see something the
further down the feed it sits.

> ⚠️ **These two caps compose, and the softer one binds first.** Position decay
> alone passes 6.5 items. A budget above that is not a constraint at all — at
> the default of 30 it removes 1% of what decay already let through, and at 60,
> **0.01%**. Sweeping a budget across 30/60/120 compares three identical
> platforms. This cost a full 10-seed study before it was caught.

<details>
<summary><b>▸ The math: candidates, ranking, attention</b></summary>

**Candidates** — `C_p = followers(author_p) ∪ inject(p, k_inj)`, vectorised as
a ragged gather over the graph's compressed-column structure rather than a
per-user Python loop. A `fanout_cap` (400) bounds how many followers one post
reaches per tick.

**Affinity ranker** — `score = a_u[topic_p] − ‖s_u − s_p‖`.
**Engagement-optimized** — that, plus `arousal_p + log(1 + engagement_p)`.

**Attention:**

```
B_u          ~ Poisson(attention_budget · activity_u)
P(see at rank r) = exp(−r / τ_pos) · ρ^depth
```

An item is seen if it is within the user's budget **and** survives the
visibility draw. `ρ^depth` (`cascade_depth_decay`, 0.7) additionally shrinks
the reach of a post far from its cascade root.

Measured share of position-decay survivors that the budget *additionally*
removes, at τ_pos = 6:

| budget | 3 | 10 | 15 | 30 | 60 | 120 |
|---|---|---|---|---|---|---|
| binds on | 63% | 22% | 10.0% | 1.00% | 0.01% | 0.00% |

(Position decay alone passes `Σ_r exp(−r/τ) = 6.51` items.)

Vary `tau_position` to ration attention, or take the budget below ~15 where it
starts to bite. `tests/test_attention_budget_binds.py` pins this.

</details>

---

## 8. The engagement kernel: the theory of *why* people react

A user is shown a post. They do one of six things: **skip, like, reply, repost,
quote, report**. The rule that decides is called the **kernel**, and it is where
a theory of online behaviour gets written down.

Each candidate action accumulates a **utility** from the features of the
situation — how much the user agrees, how arousing the post is, how popular it
already is — and the action taken is drawn from those utilities. Skipping is the
default; a feature has to earn a reaction.

**Five kernels ship, and swapping between them is swapping theories:**

| Kernel | The claim it encodes |
|---|---|
| `homophily` | people engage with what they agree with |
| `outrage` | people engage with what angers them, more so if contrarian |
| `bandwagon` | people engage with what is already popular |
| `epistemic` | people engage with what is true and new |
| `null` | people engage at a fixed base rate, blind to content |

`null` is not a throwaway. It is the **control**: same population, same graph,
same activity, no content sensitivity. Every reported effect is measured against
it, because heavy-tailed activity alone manufactures most of what naively looks
like emergent structure.

<details>
<summary><b>▸ The math: multinomial logit</b></summary>

```
U_a(u, p) = θ_aᵀ · φ(x_u, d_p, ctx)      a ∈ {like, reply, repost, quote, report}
U_skip    = 0
P(a | u, p) = exp(U_a) / (1 + Σ_a' exp(U_a'))
```

`skip` is the **reference category**, fixed at zero utility. Only differences
from skipping are identified, which is what makes the θ interpretable.

**The feature map φ** is the theory; θ is its parameters. Swapping theories
means swapping φ and θ, and nothing else in the codebase changes.

| Feature | Definition |
|---|---|
| `affinity` | `a_u[topic_p]` |
| `agreement` | `−‖s_u − s_p‖ / √D` |
| `arousal`, `novelty`, `specificity`, `quality` | the post's own dims |
| `arousal_x_neu` | arousal × neuroticism |
| `provoc_x_con` | provocativeness × contrarianism |
| `disagree_x_con` | `−agreement × contrarianism` |
| `prominence` | `log(1 + prominence_author)` |
| `social_proof` | `log(1 + engagement_count_p)` |
| `tie_strength` | 1 if the viewer follows the author |
| `recency` | `−(t − t_p)` |
| `credulity_x_q` | `credulity_u × (1 − specificity_p)` |

`disagree_x_con` exists because a kernel linear in a single global θ cannot
express "disagreement raises engagement, *more so for contrarian users*" from
`agreement` alone. Outrage needs that interaction.

**Per-action intercepts are load-bearing.** Without them every action starts at
U = 0, so `P(skip) = 1/6 = 17%` and the simulation engages on 83% of exposures
*no matter what the kernel is* — even `null`. Real platforms sit at a few
percent, and the difference propagates everywhere: an 80% engagement rate drove
the cascade reproduction number to ~16 against a requirement of < 1.

| action | intercept |
|---|---|
| like | −3.5 |
| repost | −7.0 |
| quote | −8.0 |
| report | −8.0 |
| reply | −8.5 |

Levels are set so a featureless exposure engages ~4% of the time, with roughly
65% like / 14% repost / 14% reply / 5% quote / <1% report.

**Why `agreement` is divided by √D.** The raw Euclidean distance has a mean
that grows like √D (−1.13 at D=1, −2.26 at D=3, −3.01 at D=5) while its spread
barely moves. Raising D would silently subtract a constant from every utility —
a dimensionality-dependent intercept shift wearing a feature's clothes. Dividing
by √D keeps the feature's location and scale fixed, so a θ authored at one
dimensionality means the same thing at another. (`agreement_metric = "euclidean"`
restores the raw distance if you want it.)

</details>

---

## 9. Cascades

A repost or quote creates a **new post** that inherits the original's root and
re-enters the feed with the resharer as author. That is how something spreads
beyond its author's followers.

Reposts are verbatim; quotes move the stance 40% of the way toward the quoter's
own position, because a quote carries the quoter's framing.

The system is tuned to be **subcritical with a heavy tail**: the average cascade
dies out, but the tail crosses into virality often enough to reproduce observed
cascade-size distributions. Over 90% of posts get no reshare at all.

<details>
<summary><b>▸ The math: the branching number</b></summary>

```
R = E[# reposts per exposure] · E[audience per repost]
```

Calibrated so `E[R] < 1` (cascades usually die) with `Var[R]` large enough that
the tail crosses 1. `r_eff` is logged every tick as a diagnostic. Hard caps
(`max_cascade_depth = 25`, `max_cascade_size = 1000`) raise a **warning** rather
than silently truncating — a truncated cascade that says nothing is a corrupted
measurement.

Note that thread depth comes from the reply process ([§6](#6-replies-and-threads)),
not from reposts. The two are separate branching processes with separate
critical points, and spec §5.1 requires both `>90% singletons` **and**
`depth 1.5–3` — which can only both hold if roots branch rarely while threads
already started continue often. A single flat branching probability *p* gives
depth `1/(1−p)`, which is 1.1 at p = 0.09 no matter how anything else is set.

</details>

---

## 10. The public mood

The platform has a memory of its own, independent of any user. Two quantities
carry it:

- **`s(t)`** — the **agenda**: how much attention each topic is getting.
- **`σ(t)`** — the **dominant position**: for each topic, the stance currently
  winning on it.

Both feed back into what people post ([§5](#5-what-a-post-is)) and are updated
by **engagement, not post count**. That asymmetry is the point: it is what lets
a small number of highly-engaged users capture the agenda.

<details>
<summary><b>▸ The math</b></summary>

```
s(t+1)    = ρ_s · s(t) + (1 − ρ_s) · normalize( Σ_p w_p · onehot(topic_p) )
σ(t+1)[k] = ρ_σ · σ(t)[k] + (1 − ρ_σ) · weighted_mean( stance_p : topic_p = k )
w_p       = engagement_count_p
```

`ρ_s = ρ_σ = 0.9`: exponential decay with a memory of roughly ten ticks. The
decay runs **unconditionally**, even on a tick that produced no engagement, so a
quiet period lets attention fade rather than freezing the agenda.

</details>

---

## 11. Drift: how people change

Users are not fixed. Two channels move them, plus a spring that pulls them back.

1. **Reinforcement** — you drift toward whatever style got you *more engagement
   than you expected*. Not more engagement in absolute terms: more than your own
   running baseline, so a big account is not permanently reinforced for being
   big. This channel moves expression traits, and — at its own rate,
   `drift_lr_behavior` — the behaviour propensities with a generation-map path
   (`activity`, `reply_prop`, `repost_prop`), because outrage expression is
   socially *learned* (Brady et al. 2021), not a fixed disposition. Engagement
   *choice* traits (`contrarianism`, `credulity`) have no residual to learn
   from and are left to the learnable kernel ([§16](#16-change-spec-mechanisms)).
2. **Social influence** — your stance drifts toward the content you consumed and
   did not reject. Liking and reposting pull you toward a post; replying pulls
   slightly away; reporting pushes away hard. The weight table is ablatable:
   `repulsion=False` zeroes the negative half, which is Mäs & Flache (2013)'s
   open question, not a settled assumption ([§16](#16-change-spec-mechanisms)).
3. **Mean reversion** — every trait is pulled back toward a slow-moving personal
   baseline, at a rate that differs by block: style is fashion and reverts fast;
   stance reverts slowly; personality effectively not at all.
4. **Affect** — a separate state block (identification with your camp, animus
   toward the other) that updates from interaction outcomes
   ([§16](#16-change-spec-mechanisms)); off unless `population.affect` is set.

Drift gains **ramp linearly from zero** over the first 50 ticks, so switching it
on does not jolt the population.

<details>
<summary><b>▸ The math</b></summary>

All of it operates on `X_stored` (unconstrained), so it is plain addition and
can never leave the feasible set.

**Channel 1 — reinforcement.** With surprise `r_p = engagement_p −
E[engagement | author_p]` (the expectation being a per-user EMA):

```
residual = actual_style_stored − A · x_stored[author]
Δ_expr   = lr · mean_over_posts( r_p · (residual · A_expr) )
```

`A_exprᵀ · residual` is exactly the gradient of the linear expression map — the
direction in trait space that would have produced more of whatever got engaged
with. **Averaged**, not summed, over an author's posts this tick, so the step
size is set by `drift_lr` rather than incidentally scaling with how often
someone happened to post.

**Channel 2 — social influence.**

```
Δ_stance = lr_social · mean_over_exposures( w_action · (s_p − s_u) )
```

| action | like | repost | quote | reply | report | skip |
|---|---|---|---|---|---|---|
| weight | +1.0 | +1.5 | +0.5 | −0.5 | −2.0 | 0 |

**Mean reversion** — Ornstein–Uhlenbeck toward a slow baseline `Bs`, itself
initialised to the population's own starting traits (not zero):

| block | personality | expression | topic_affinity | behavior | meta | stance | affect *(C1)* |
|---|---|---|---|---|---|---|---|
| rate *k* | 0.00 | 0.05 | 0.02 | 0.01 | 0.01 | 0.005 | 0.02 |

Plus Gaussian noise at `noise_sigma = 0.002`.

**Channel 3** (LLM adjudication of a salient event) is **queued, never
executed inside the tick** — events are logged for an offline pass. No network
call ever happens inside the loop.

</details>

---

## 12. One tick, start to finish

```
 1. decay the public mood                    s, σ ← ρ·(…)
 2. draw who posts               → new posts (topic, stance, style)
 3. draw replies from thread heat → reply posts        [Hawkes, §6]
 4. build each user's candidate set   followers ∪ injected
 5. rank the candidates                                [the ranker, §7]
 6. cap by attention budget and position decay → exposures
 7. draw an action per exposure                        [the kernel, §8]
 8. reposts/quotes create derived posts                [cascades, §9]
 9. update the public mood from this tick's engagement
10. drift traits                                       [§11]
11. queue salient events for the offline language pass
12. retire posts older than post_lifetime
```

Steps 9–11 run **outside** the "was anyone exposed?" guard: a quiet tick still
decays the agenda and still drifts.

Every phase draws from its **own named random stream** (`timing`, `generation`,
`exposure`, `reaction`, `population`), derived from the single seed. Changing
the number of exposures therefore does not shift the timing draws, so one
mechanism can be altered without reshuffling every other.

---

## 13. Measuring it

### Validity gate: does it behave like a platform?

Eight **stylized facts** with target ranges taken from empirical literature.
These are a *gate*, not a result — they say whether the model behaves enough
like a platform to reason from.

| Fact | Target |
|---|---|
| Engagement per post (power-law α) | 2 – 3 |
| Cascade size (share of singletons) | ≥ 90% |
| Thread depth (mean, branched) | 1.5 – 3 |
| Attention Gini (lifetime, per post) | 0.80 – 0.95 |
| Posting volume Gini | 0.70 – 0.90 |
| Reciprocity | 0.20 – 0.40 |
| Clustering vs degree-matched null | ≥ 3× |
| Inter-cluster interaction rate | ≤ 0.33 |

Two are currently missed, for a reason that is understood and documented rather
than tuned around: attention concentration is capped by the graph generator, not
by the kernel ([§3](#3-the-network)).

### The result: six normative outcomes

| Outcome | The question |
|---|---|
| `cross_cutting_exposure` | are people put in front of the other side? |
| `voice_inequality` | who gets heard — including whether the minority camp is heard at all? |
| `epistemic_alignment` | does merit predict attention? |
| `hostility_given_contact` | when camps do meet, how badly does it go? |
| `feed_narrowing` | how much narrower was the feed than the world it was drawn from? |
| `selection_filtering` | what did *choice* filter that the algorithm did not? (C2: the echo-chamber index computed on attended exposures minus the same index on everything exposed — the Bakshy decomposition's choice component) |

`hostility_given_contact` deliberately returns the **contact rate and the
hostility rate together**, because a platform that eliminates cross-camp contact
trivially eliminates cross-camp hostility, and reporting only the second would
score that as a success.

### The matched null, and two contrasts that are easy to confuse

Every cell is run twice: once with the real kernel, once with `kernel="null"`,
**same population, same graph, same seed**. That gives two different numbers:

- **`lever_effect`** — the outcome minus the outcome at the lever's *reference*
  value. *What moving this dial does.* **This is the headline.**
- **`kernel_delta`** — model minus its matched null at the *same* lever setting.
  *Was this mediated by the engagement kernel?* Near zero for feed levers **by
  construction**, because the null holds the lever fixed too.

Reporting `kernel_delta` as "the effect of the ranker" is the easy mistake and
reads as ≈ 0 for every feed lever.

### One lever at a time is a screen, not a study

The sweep varies one design choice at a time, which means every lever is
evaluated **at the base configuration**. A flat row means "flat there" — a
weaker claim than "this does not matter", and the difference is not academic:
`tau_position` reads flat in the screen and moves cross-camp exposure by
**+0.088 under `affinity` against −0.004 under `chronological`**.

`build_interventions(..., across=)` crosses each lever against a background
factor and `interaction_table` reports the **swing** across backgrounds. Results
are in [`FINDINGS.md`](FINDINGS.md).

### Reproducibility

A run is identified by the **structural hash of its config** plus its seed, and
cached at `dlab/runs/{hash}/{seed}/`. The hash is computed from canonical JSON —
sorted keys, fixed float formatting — so it depends on exactly the values that
determine the artifact and nothing else. Change any config field and you get a
different run; change none and `cached_run` returns the existing one.

---

## 14. Every knob, in one table

### Population

| Field | Default | What it does |
|---|---|---|
| `n_users` | 10000 | population size |
| `n_topics` | 8 | number of subjects |
| `stance_dims` | 3 | *D*, the dimensionality of the political space |
| `activity_sigma` | 1.8 | spread of posting rates. **This parameter *is* the posting-Gini target** — Gini of a lognormal is `erf(σ/2)` in closed form |
| `pareto_alpha` | 2.3 | tail of the prominence distribution |
| `topic_logit_sigma` | 1.0 | spread of topic interest |
| `archetype_weights` / `_offsets` | library defaults | the named groups and their trait shifts |
| `correlation_pairs` | () | requested trait correlations — **adds to** what archetypes already induce |
| `animus_mu` | -2.2 | log-space mean of the initial `animus` draw (needs `affect=True`) — Experiment 03's affective-tribalization dial |
| `stance_polarization` | 0.0 | axis-0 stance drawn from `bimodal_normal(separation=this)` instead of `normal(0,1)` above 0 — Experiment 03's ideological-tribalization dial, continuous unimodal → strongly bimodal |

### Graph

| Field | Default | What it does |
|---|---|---|
| `generator` | `latent_space` | which model builds the network |
| `mean_degree` | 40 | average number followed |
| `homophily_beta` | 0.35 | strength of "follow people like me" |
| `prominence_gamma` | 0.6 | strength of "follow people others follow" |
| `long_tie_fraction` | 0.1 | share of ties drawn at random — the shortcuts |
| `knn_k` | 60 | candidate pool size; **must exceed `mean_degree`** |
| `mirror_p` | 0.02 | probability each edge is mirrored — *not* the measured reciprocity |
| `fanout_cap` | 400 | max followers one post reaches per tick |
| `sbm_block_source` | `archetype` | `sbm` generator only: `archetype` \| `topic_affinity` \| `camp` (blocks = sign of the dominant stance axis — Experiment 03's structural-tribalization dial; see `network.measures.cross_camp_tie_share`) |
| `sbm_homophily` | 0.8 | `sbm` generator only: between:within edge-probability ratio, smaller = more sorted — the dial itself, at whichever `sbm_block_source` is chosen |

### Dynamics — timing and volume

| Field | Default | What it does |
|---|---|---|
| `n_ticks` | 500 | how long to run |
| `ticks_per_day` | 24 | ticks per day; sets the circadian period |
| `posts_per_tick_rate` | 0.02 | posting rate at `activity = 1` |
| `fatigue_decay` | 0.9 | how fast a posting burst wears off |
| `post_lifetime` | 5 | ticks a post stays in feeds |

### Dynamics — the feed

| Field | Default | What it does |
|---|---|---|
| `ranker` | `chronological` | **the central design lever** |
| `inject_k` | 0 | random non-followers reached per post |
| `attention_budget` | 30 | items per tick — **inert above ~15; see §7** |
| `tau_position` | 6 | how far down the feed people read — **the lever that actually rations attention** |

### Dynamics — engagement

| Field | Default | What it does |
|---|---|---|
| `kernel` | `homophily` | **the theory of engagement** |
| `kernel_theta` | () | override the kernel's weights |
| `agreement_metric` | `rms` | divide distance by √D so θ transfers across D |

### Dynamics — replies and cascades

| Field | Default | What it does |
|---|---|---|
| `hawkes_mu0` | 0.004 | baseline reply intensity |
| `hawkes_ratio` | 0.6 | α/β; **must stay < 1** |
| `hawkes_beta` | 1.5 | how fast thread heat decays |
| `hawkes_mu_inherit` | 1.0 | heat a reply's own thread inherits — **the depth dial; see §6** |
| `max_replies_per_tick` | 1 | 1 = chains not bushes; 0 = uncapped |
| `max_thread_age` | 15 | ticks a thread stays open |
| `cascade_depth_decay` | 0.7 | ρ in `ρ^depth` reach decay |
| `max_cascade_depth` / `_size` | 25 / 1000 | caps that **warn**, not truncate silently |

### Dynamics — mood and drift

| Field | Default | What it does |
|---|---|---|
| `trend_eta` | 0.3 | susceptibility to what is trending |
| `rho_s` / `rho_sigma` | 0.9 | memory of the agenda / dominant stance |
| `drift` | `full` | `none` \| `social` \| `full` |
| `drift_lr` / `drift_lr_social` | 0.02 / 0.01 | step sizes for the two channels |
| `drift_lr_behavior` | 0.01 | reinforcement rate for behavior propensities (C3a) |
| `drift_ramp_ticks` | 50 | linear ramp-in |
| `ou_k` | () | per-block mean-reversion overrides |
| `noise_sigma` | 0.002 | random walk on traits |

### Dynamics — change-spec mechanisms (§16)

| Field | Default | What it does |
|---|---|---|
| `affect` *(population)* | False | the affect block: `identification` + `animus`, updating from interaction outcomes (C1) |
| `lr_affect` / `affect_ou_k` | 0.015 / 0.02 | affect step size / reversion — **the reversion rate is a guess** |
| `affect_weights_hostility` | see `drift.py` | C1.3's per-action MAGNITUDE only as of V2 (every entry >= 0; `report` removed) |
| `affect_weights_support` | see `drift.py` | C1.3's identification table — untouched by V2, action-keyed only |
| `affect_valence_signs` | see `drift.py` | V2: the (action, valence) table's SIGN, keyed on the four `{agree,disagree}_{civil,hostile}` cells |
| `affect_drive` | `distance` | V7.3: `camp` (pre-V7.3 binary, gated on Sarle bimodality) \| `distance` (continuous `phi(d)`, no gate — default since V7.6's re-gate passed) |
| `affect_d0` | 1.0 | V7.3's `phi(d) = d / (d + affect_d0)` saturation constant; also V7.4's `d_cross` cross-contact threshold |
| `civility_prob` / `force_agree` / `force_civil` | 0.5 / None / None | V1's exogenous valence: civil/hostile coin flip; force_* pin an axis for fixtures |
| `valence_mode` | `exogenous` | V1/V2's coin flip, or `endogenous` (V3: per-user logits on animus/identification/distance) |
| `valence_beta0/_beta_dist/_beta_ident/_noise_agree` | 0 / -1 / -0.3 / 1 | V3's P(agree) logit — see `EndogenousValenceParams` for the fixed signs |
| `valence_gamma0/_gamma_animus/_gamma_dist/_noise_civil` | 0 / -1 / 0 / 1 | V3's P(civil) logit — `gamma_animus` is the self-reinforcement term the bistability probe needs |
| `report_exit` / `report_animus_increment` | False / 0.0 | V4: a report suppresses future exposure to that author; optional small direct animus effect |
| `selection` | `position_only` | content-conditional attention: `homophilous` \| `arousal_seeking` (C2) |
| `selection_beta` | () | overrides for the selection logit's betas |
| `rewire` / `rewire_every` / `rewire_rate` | False / 25 / 0.01 | slow follow/unfollow on accumulated interaction valence (C2) |
| `kernel_learning` / `_rule` | `none` / `conformity` | learnable kernel gains: `group_gain` \| `full`; rule `conformity` \| `bandit` \| `habituation` (C3b) |
| `lr_kernel` / `kernel_gain_ou_k` | 0.01 / 0.02 | gain step size / reversion toward the anchored kernel |
| `quality_trait_coupling` | 0.0 | 0 = quality independent of author traits (C4); 1 = the old confounded map |
| `social_weights_positive` / `_negative` | see §11 | channel-2 weight halves |
| `repulsion` | True | False zeroes the negative weights — the ablation, not a clamp (C6) |
| `silence_gate` / `_conviction_moderation` | 0.0 / 1.0 | spiral-of-silence expression gate on perceived climate (C7) |
| `reply_model` | `hawkes` | `hawkes` (simple contagion) \| `threshold` (distinct engaged neighbours, C8) |
| `threshold_scale` | 16.0 | threshold response curve — needs its own §5.1 calibration before D5 |

### Recording

| Field | Default | What it does |
|---|---|---|
| `snapshot_every` | 1 | ticks between trait snapshots |
| `exposure_sample_rate` | 0.01 | share of exposures logged — exposures outnumber engagements ~50:1, so only a sample is kept |

> ⚠️ At 1%, a user contributes ~2 rows per run, so **only population means are
> interpretable**, never a per-user value. Raising it changes the config hash and
> forks the cache — pick one rate up front and use it everywhere.

---

## 15. What this model cannot tell you

Stated plainly, because a model's limits are part of its specification.

- **No deliberation, no persuasion by reason.** Drift is social influence and
  reinforcement only. Nobody is argued out of a position by a better argument.
  The model speaks to *structural preconditions* for democratic discourse — who
  is exposed to whom, who is heard — **not** to deliberative quality.
- **`quality` is generated, not evaluated.** It comes from author traits, so
  `epistemic_alignment` is interpretable only as a difference from the null.
  *(C4 closed the worst part of this: `quality_trait_coupling=0` — now the
  default — draws quality independently of author traits, and
  `quality_attention_lift` refuses to compute without a matched null run.)*
- **Camps are a statistical split, not groups.** "Camp" is the sign of a user's
  position on the dominant axis of stance variation. It is defined even when the
  population is a single unimodal blob, where it is noise. The narrator gates
  camp language on a bimodality test (Sarle's coefficient > 5/9); analyses using
  camps should say whether the population is actually bimodal. *(C1's affect
  machinery inherits this gate: where camps are undefined, animus comparisons
  are reported as undefined, not zero.)*
- **Attention concentration is capped by the graph generator.** Two of the eight
  stylized facts are missed for this reason. It is a known, located limitation,
  not a mystery. *(Partially addressed by the C9 gate: with `latent_pa` +
  `engagement_optimized` + `bandwagon`, attention Gini and reciprocity hold
  their spec ranges simultaneously — see `tests/test_change_spec.py::test_c9_*`.
  The screen's base configuration does not use that combination, so its own
  rows still read as before.)*
- **The affect block's reversion rate is a guess.** `affect_ou_k = 0.02`
  ("stickier than style, less sticky than position") has no empirical anchor;
  it is carried in the C10 sensitivity sweep (`experiments/sensitivity_sobol.py`)
  and any affect result should be reported with its Sobol indices attached.
- **The screen is not the study.** A lever that reads flat has been shown flat
  *at the base configuration*. See [§13](#13-measuring-it).
- **It is not calibrated to any specific platform.** The targets come from
  general empirical literature. Absolute numbers are not predictions; the
  *comparisons between configurations* are the output.

---

## 16. Change-spec mechanisms

Ten literature-informed mechanisms (discourse-lab-changes.md, C1–C10), all
**defaulted off** so nothing below this point moves unless asked. Each is a
registered component or a named config field, and each is covered by a
conformance test that observes its *effect* rather than its definition
(`tests/test_change_spec.py`).

| Mechanism | Warrant | Dial | Default |
|---|---|---|---|
| **C1 affect block** | Affective polarization — animus toward the other camp, attachment to one's own — rose while ideological positions moved little (Iyengar, Sood & Lelkes 2012; Iyengar & Westwood 2015); out-group animus is the strongest single engagement predictor measured (Rathje et al. 2021) | `population.affect`, `lr_affect`, `affect_ou_k` | off |
| **C2 selection** | Individual choice filtered cross-cutting content *more than the algorithm* (Bakshy, Messing & Adamic 2015) — a stage the model lacked entirely | `dynamics.selection`, `selection_beta` | `position_only` |
| **C2 rewiring** | Sorting into echo chambers through follow/unfollow (Törnberg, PNAS 2022) | `rewire`, `rewire_every`, `rewire_rate` | off |
| **C3 reinforcement** | Outrage expression is socially learned (Brady, McLoughlin, Doan & Crockett 2021) — fixed-disposition kernels cannot produce norm convergence | `drift_lr_behavior`, `kernel_learning`, `kernel_learning_rule` | off |
| **C4 quality backdoor** | `quality` was generated from author traits, confounding Spearman(quality, engagement) (Salganik & Muchnik's condition: merit independent of the artist) | `quality_trait_coupling` | **0.0** (independent draws) |
| **C5 out-group attraction** | Out-group content raised sharing odds 67% — general and large, not a minority-trait effect (Rathje et al. 2021) | `outgroup` feature in the kernel | on (feature) |
| **C6 repulsion ablation** | The repulsive-influence assumption has mixed, hard-to-identify support (Mäs & Flache 2013; Takács et al. 2016) | `repulsion=False` zeroes the negative channel-2 weights | on |
| **C7 spiral of silence** | Perceived network disagreement predicts self-censorship (Hampton et al., Pew 2014; Matthes et al. 2018) | `silence_gate`, `silence_conviction_moderation` | 0 (off) |
| **C8 complex contagion** | Behaviours spread through reinforcing exposures from *distinct* neighbours; long ties slow complex contagion, inverting Granovetter (Centola & Macy 2007; Centola 2010) | `dynamics.reply_model` | `hawkes` |
| **C9 attention cap** | 10% of users produced 97% of political tweets (Pew 2019); the gate demands Gini **and** reciprocity in range at once | `latent_pa` generator + the stylized gate test | generator opt-in |
| **C10 harness** | Equifinality: many mechanisms produce the same macro pattern (Grimm et al.) | `experiments/identify.py`, `sensitivity_sobol.py`, `designs.py` | — |

### V1-V6: engagement valence and the de-escalation channel

Six more mechanisms ("Change Spec V1 — Engagement Valence and the
De-escalation Channel"), same discipline: defaulted to the pre-existing
behaviour, each covered by a conformance test
(`tests/test_change_spec.py::test_v1_*`-`test_v6_*`). Numbered separately
from C1-C10 because they change the affect channel's *form* (what a
hostility increment is keyed on), not its parameters.

| Mechanism | Warrant | Dial | Default |
|---|---|---|---|
| **V1 engagement valence** | A supportive cross-camp reply and a quote-dunk were the same event (both just "reply") — no representation for conduct independent of position | `dynamics.civility_prob`, `force_agree`/`force_civil` | civility_prob 0.5 (exogenous) |
| **V2 (action, valence) table** | The hostility table's only sign was decided by action, so no override could express contact *reducing* hostility without making every instance of that action de-escalating, civil or hostile alike | `affect_weights_hostility` (magnitude only), `affect_valence_signs` (sign) | see `drift.py` |
| **V3 endogenous valence** | Exogenous valence cannot produce basins; the question — does a population already hostile metabolize contact as attack — needs the feedback loop | `dynamics.valence_mode`, `valence_beta*`/`valence_gamma*` | `exogenous` (V1/V2's coin flip) |
| **V4 report as exit** | Reporting is *dis*engagement (the user exits), not the largest hostility increment — reverse causation compiled forward | `dynamics.report_exit`, `report_animus_increment` | off |
| **V6(2) emergent k** | `camps_and_bimodality` hardcodes k=2 before camps are defined; an intervention that fragments two camps into five hostile ones reads as "bimodality fell" under a binary frame | `metrics.polarization.emergent_camps` | measurement only, not wired into dynamics |

`V6(1)` (continuous per-axis distance in the mechanism, not camp
membership) needed no dial — the kernel's `agreement` feature and V1's
`agree_delta` threshold already operate on the full stance vector; the
requirement is a regression guard
(`test_v6_1_agreement_is_continuous_per_axis_not_camp_membership`), not new
code. `V5` is the conformance-test discipline itself (this section's own
test files), not a runtime mechanism — written first, per the spec's own
sequencing, so that four of its five assertions were red before V1 landed.
V6's group-directed (vector) animus is explicitly deferred in the spec
itself — scalar animus stays "generalized out-group hostility" for now.

### V7: continuous affect drive

One mechanism ("Change Spec V7 — Continuous Affect Drive"), diagnosing and
then closing the coupling V1-V6 left standing: the C1.3 affect op's
`camps is not None` gate froze animus/identification for every arm below
the Sarle bimodality threshold, not only for the camp-conditional kernel
features — so "does an intervention change affective hostility" was not
merely hard to detect below the gate, it was not measurable at all.

| Mechanism | Warrant | Dial | Default |
|---|---|---|---|
| **V7.1 gate instrumentation** | Nothing recorded whether the affect op was skipped on a given tick; a run's own consistency could be the signature of a bimodality-gate switch as much as of a genuine dose-response | `metrics.parquet` columns `bimodality`, `affect_gated`; `dynamics.drift.affect_gate_active` | always on (instrumentation, no behaviour change) |
| **V7.3 continuous drive** | The camp label is a binary median-split projection artifact below the gate; V6(1) already moved the kernel's own agreement feature onto continuous per-axis stance distance, and the affect op was the last consumer of the camp label left in the mechanism path | `dynamics.affect_drive` (`camp`\|`distance`), `affect_d0` | **distance** (flipped from `camp` once V7.6's re-gate passed — FINDINGS.md) |
| **V7.5 BIC margin** | `delta_k` is a step function — a population drifting steadily toward a k-change is indistinguishable from a flat null until the argmax flips | `metrics.polarization.emergent_camps`'s `bic_margin` | measurement only, not wired into dynamics |

V7.2 and V7.4 are analysis-only additions to `experiments/
experiment03_bubble_intervention.py` (retaining the `none` arm's own
absolute ideological movement as `ideo_level_*`, and a cross-camp-restricted
contact denominator for `delta_aff`) — no dial, since neither touches a
mechanism. V7.6 re-gated Experiment 03's own substrate under the new
mechanism (the first time it had been gated at all — see FINDINGS.md) and
demoted `attention_gini` out of the blocking `GATE_ROWS` into a reported
diagnostic: its [0.8, 0.95] band is reachable only under the `bandwagon`
kernel, and Experiment 03 runs `outrage`.

### The affect block, in one equation

Two new trait columns — `identification` (attachment to own camp, logit) and
`animus` (hostility toward the other, log) — sampled correlated with
`conviction` and `contrarianism` respectively. **Camp** is the sign on the
dominant axis of stance variation, gated on Sarle's bimodality > 5/9: where
the population is unimodal, camps are noise and every affect number is
reported as undefined, not zero. Affect is a *state* — it updates from
interaction outcomes, with weight tables deliberately separate from social
influence's (replying to out-group content is *engagement* with it, so it
raises animus while being stance-repulsive):

```
Δanimus_u         = lr_affect · mean_over_exposures( outgroup · hostility_weight(action) · valence_sign(cell) )
Δidentification_u = lr_affect · mean_over_exposures( ingroup  · support_weight(action) )
```

`outgroup`/`ingroup` are `dynamics.affect_drive`-dependent (V7.3): the binary
camp label above under `"camp"`, or `phi(d(s_i, s_j)) = d / (d + affect_d0)`
and its complement under the default `"distance"` — a saturating function of
the dyad's continuous per-axis stance distance that needs no camp label and
so never gates on bimodality at all.

The `valence_sign(cell)` factor is V2 (below): pre-V2, `hostility_weight`
alone was always >= 0, so animus was monotone non-decreasing and no
configuration could represent contact *reducing* hostility (recorded in
FINDINGS.md). `identification` did not gain a valence factor — the change
spec's re-keying is scoped to the hostility table alone.

### Learnable kernels, three tiers

`kernel_learning` modulates a named kernel and never replaces it — a run must
stay describable as "outrage, plus this much learned deviation".

| Tier | State per user | Reading |
|---|---|---|
| `none` | — | theta is a table you can read |
| `group_gain` | one gain per named coefficient group (agreement, outgroup, arousal, social_proof, recency, affinity) | "this user weights the out-group group 1.4× the norm" |
| `full` | coefficient-granular gains | unrestricted θ_u; honesty about the toolbox claim |

The learning rule is **named in the config, never implied**:
`conformity` (gains drift toward in-neighbours' *revealed engagement
behaviour* — Brady's norm convergence),
`bandit` (own posts' above-baseline engagement), and `habituation` (use-driven,
no social channel — the control, standing to conformity as `null` stands to
the kernels). Gains revert OU-style toward 1, so a learned kernel cannot
wander off and quietly stop being the theory it names.

### Selection, silence, contagion

**Selection** sits between ranking and the kernel:
`P(attend | exposed) = σ(β_pos·pos_decay + β_agree·agreement + β_arousal·arousal)`.
The exposure log persists both `exposed` and `attended`, so
`selection_filtering.selection_shift` measures the choice component of the
Bakshy decomposition without conflating it with the ranker's.

**The silence gate** multiplies posting probability by a perceived-climate
factor — perceived through the feed blend of [§12](#12-one-tick-start-to-finish),
not the global state, and moderated by conviction. It gates *whether one
posts*, never what a post says, which is what makes false consensus
(`expressed_vs_latent_bimodality` < 0) reachable.

**The threshold reply model** replaces the Hawkes intensity draw with
propensity rising super-linearly in the count of *distinct* engaged
in-neighbours. The separating experiment is the crossover with graph
structure (`d5_contagion_crossover` in `experiments/designs.py`): hawkes
spreads faster across long ties, threshold inside high clustering. If both
respond identically to clustering, `threshold` is not actually complex.

### The C10 harness

`experiments/identify.py` reports whether two theories are separable **on the
metrics the model reports** — AUC plus which metric carries the signal. AUC ≈
0.5 is a reportable result, not a failure. `experiments/designs.py` holds the
discriminating experiments as runnable designs, each required to state its
falsifier; `experiments/sensitivity_sobol.py` attaches Sobol indices to every
result driven by a parameter with no empirical anchor (`lr_affect`,
`affect_ou_k`, `silence_gate`, …).

### Experiment 03 infrastructure: time-varying configs and forked arms

`DynamicsConfig.schedule` is a piecewise list of `(start_tick, overrides)`
pairs: at tick `t`, `config.py::effective_dynamics` applies the latest
entry whose `start_tick <= t` on top of the base dynamics config, entries
NOT cumulative (a later entry restates values directly rather than undoing
an earlier one). Every other phase of a tick — including RNG draws, keyed
on `seed` alone and never on config content — is unaffected by which
schedule fires, so two configs sharing a seed and an identical dynamics
config up to some tick `T`, differing only in a schedule entry AT `T`,
produce a **bit-identical** per-tick record for every `t < T`
(`tests/test_runner.py::test_schedule_gives_a_bit_identical_prefix_and_diverges_after`).
This is what lets an experiment fork several "arms" from one shared burn-in
without the confound of separate configs that merely start out the same.

Only fields `TickEngine.step` reads off its per-tick local — or rebuilds
from it, as the eight `valence_*` coefficients now are — are
schedule-reactive; a field consumed once at construction (`kernel_learning`,
`quality_trait_coupling`, `agreement_metric`'s calibration) is not, and
scheduling it silently does nothing past tick 0. `experiments/
experiment03_bubble_intervention.py` is the first consumer: `forked_config`
builds one config per arm from a shared `burn_in` config, each a schedule
entry at the intervention tick (plus, optionally, a second entry at a
withdrawal tick that restates the `none` arm's values — the hysteresis
phase of that experiment's design).

---

## Where to go next

| | |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | measured results and negative results |
| [`notebooks/demo.ipynb`](notebooks/demo.ipynb) | the API, runnable, in build order |
| [`discourse-lab-spec.md`](discourse-lab-spec.md) | the formal specification |
| [`discourse-lab-dev.md`](discourse-lab-dev.md) | design decisions and their rationale |
