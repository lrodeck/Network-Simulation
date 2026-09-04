# Discourse Lab — Specification

A vectorized, non-agentic simulation toolbox for experimenting with theories of
large-scale online discourse. LLMs are used only to realize text from
already-decided numeric state, never to make decisions inside the dynamics loop.

---

## 0. Design principles

1. **Dynamics are numeric; language is a rendering pass.** The simulation is a
   sequence of matrix operations. Text generation is an offline pass over a
   completed run. Experiments on dynamics never call an API.
2. **Every modeling assumption is a swappable component.** Population sampler,
   graph generator, feed ranker, engagement kernel, action model, drift model.
   Swapping a theory means passing a different function, never editing the loop.
3. **A run is a pure function of `(Config, seed)`.** Reproducibility is
   structural, not a discipline.
4. **Null models are mandatory.** Heavy-tailed activity alone manufactures most
   of what naively looks like emergent structure.
5. **Struct-of-arrays, not objects.** No per-user Python loops anywhere in the
   tick.

---

## 1. State space

### 1.1 Users

Population of `N` users. All traits live in a single matrix `X ∈ R^{N×n}`,
column-blocked:

| Block | Symbol | Dims | Notes |
|---|---|---|---|
| Personality | `x^per` | 5 | OCEAN, standardized |
| Expression | `x^exp` | 6 | verbosity, formality, irony, humor, profanity, emoji |
| Topic affinity | `a_u` | K | simplex or unbounded logits over K topics |
| Stance | `s_u` | D | D ≈ 3–5 latent ideological axes |
| Behavior | `x^beh` | 6 | activity, reply_prop, repost_prop, contrarianism, credulity, prominence |
| Meta | `x^meta` | 3 | plasticity, conviction, circadian_phase |

**Internal representation is unconstrained.** Traits bounded in `[0,1]` are
stored as logits, positive traits as logs. Constraints are enforced by the
transform on read, not by clipping. This makes drift a plain additive process
that can never leave the feasible set.

```
x_stored ∈ R^n            # what drift operates on
x_used   = link(x_stored) # sigmoid / exp / identity per column
```

Two copies are kept: `X` (current) and `B` (baseline, the mean-reversion target).

### 1.2 Posts

`M` posts accumulated over the run, dims `D_p ∈ R^{M×p}`:

```
topic       ∈ Δ^K   (or one-hot for simple runs)
stance      ∈ R^D
arousal     ∈ [0,1]     emotional activation
valence     ∈ [-1,1]    positive/negative
provocativeness ∈ [0,1] invites disagreement
novelty     ∈ [0,1]     distance from current discourse state
specificity ∈ [0,1]     concrete vs vague
quality     ∈ [0,1]     latent "merit", deliberately decoupled from engagement
length      ∈ R+
```

Plus metadata: `author`, `t`, `parent` (−1 for roots), `root`, `depth`, `kind ∈
{post, reply, repost, quote}`.

`quality` exists specifically so you can ask whether your engagement kernel
correlates attention with merit. If quality never enters the engagement score,
the answer is a controlled zero — that is a useful baseline.

### 1.3 Graph

`G ∈ {0,1}^{N×N}` sparse CSR, `G[u,v] = 1` means u follows v (u receives v's
posts). Store both CSR (followers-of) and CSC (followees-of); you need both
directions in different phases.

### 1.4 Global state

```
s(t) ∈ R^K      discourse attention over topics (decaying)
σ(t) ∈ R^{K×D}  mean stance currently dominant per topic
t               tick index
```

### 1.5 Event log

Append-only arrays:

```
engagements: (user, post, action, t)   int32
exposures:   NOT persisted — see §3.5
```

---

## 2. Mathematical pipeline

### 2.1 Population generation

Two-stage: archetype mixture over a Gaussian latent, then copula transform to
target marginals.

```
c_u ~ Categorical(π)                      archetype, |π| = C
z_u ~ N(μ_{c_u}, Σ_{c_u})   ∈ R^n         correlated latent
w_i = Φ((z_ui - μ̄_i) / σ̄_i)              probability integral transform
x_ui = F_i^{-1}(w_i)                      target marginal
```

where `μ̄, σ̄` are the *mixture* moments (not per-component), so the marginals of
each archetype are deliberately distorted relative to the population marginal.
That distortion is the point: archetypes should differ in their marginals, not
just their means.

**Marginals** `F_i`:

| Trait | Distribution | Rationale |
|---|---|---|
| activity_rate | Lognormal(μ, σ), σ ≈ 1.2 | posting volume is heavy-tailed |
| prominence | Pareto(α ≈ 2.3) | drives follower count |
| personality | Normal(0,1) | conventional |
| contrarianism | Beta(2,5) | most people are not, some strongly are |
| plasticity | Beta(2,8) | most people barely move |
| conviction | Beta(5,2) | stance is stickier than style |
| circadian_phase | VonMises(μ_tz, κ) | timezone clustering |

**Correlation structure** is specified as a sparse list of `(trait_i, trait_j, ρ)`
pairs, completed to a valid correlation matrix by nearest-PSD projection
(Higham). Do not hand-author a full `n×n` matrix; it will not be PSD and you
will not notice.

**Archetypes** are defined declaratively as offsets from population mean, e.g.

```python
Archetype("lurker",      w=0.55, offsets={"activity": -1.5, "reply_prop": -1.0})
Archetype("poster",      w=0.25, offsets={"activity": +0.8})
Archetype("firebrand",   w=0.08, offsets={"contrarianism": +1.5, "arousal_bias": +1.0})
Archetype("institution", w=0.02, offsets={"prominence": +2.5, "formality": +1.5})
Archetype("newcomer",    w=0.10, offsets={"plasticity": +1.5, "conviction": -1.0})
```

Archetype membership is retained as a label for analysis but is **never read by
the dynamics** — it only shapes the initial draw. Communities must emerge from
trait geometry, not from a group ID the model can cheat with.

### 2.2 Network generation

Default kernel — latent space with preferential attachment:

```
logit P(u → v) = α − β · d(u, v) + γ · log(1 + prominence_v)
d(u, v) = ||[s_u; a_u] − [s_v; a_v]||₂
```

`α` is calibrated by bisection to hit a target mean degree. Exact computation is
`O(N²)`; for `N > 20k` sample candidate edges via k-NN in latent space
(`sklearn.neighbors` or FAISS) plus a uniform random component for long ties.
The uniform component matters — a pure k-NN graph has no shortcuts and cascades
die in-cluster.

**Alternative generators** (same interface, swappable):
- `configuration_model` — degree sequence only, no homophily. Null model.
- `sbm` — explicit blocks, for when you want to *impose* community structure.
- `barabasi_albert` — pure preferential attachment, no homophily.
- `latent_space` — default above.

Reciprocity: after generation, with probability `r`, add the reverse edge for
each edge. Empirical `r ≈ 0.2–0.4` on follow graphs.

### 2.3 Activity and timing

Posting is an inhomogeneous Poisson process:

```
λ_u(t) = activity_u · circ(t − φ_u) · fatigue_u(t)
n_posts_u(t) ~ Poisson(λ_u(t) · Δt)
```

`circ` is a fixed diurnal shape (two-peak). `fatigue` optionally suppresses a
user after a burst.

**Replies are Hawkes, not Poisson.** Reply intensity to post `p`:

```
λ_p(t) = μ_p + Σ_{t_i < t} α · exp(−β (t − t_i))
```

This is what produces realistic thread burstiness — a post gets its comments in
a clump, not spread uniformly. Require `α/β < 1` for stability or threads run
away. This is a meaningful dial: `α/β → 1` gives you pile-on dynamics.

### 2.4 Post dimension generation

```
topic_p  ~ Categorical(softmax(a_u + η · s(t)))       η = trend susceptibility
stance_p = conviction_u · s_u + (1 − conviction_u) · σ(t)[topic_p] + ε_s
d_p      = A · x_u + B · s(t) + C · onehot(topic_p) + ε_d
```

`A ∈ R^{p×n}` is the **trait→expression map**: the single most important
authored object in the system. It encodes claims like "high neuroticism raises
arousal", "low agreeableness raises provocativeness". Author it as a sparse,
commented, named-entry table, not a dense matrix.

The stance line encodes conformity pressure: low-conviction users drift toward
whatever stance currently dominates their topic. Set `conviction = 1` globally
to disable conformity as a control condition.

### 2.5 Exposure — feed construction

This is the layer where platform design lives, and it is the most important
experimental lever in the toolbox. Three sub-steps.

**(a) Candidate inbox.** For each active post, candidates are the author's
followers, plus algorithmic injection:

```
C_p = followers(author_p) ∪ inject(p, k_inj)
```

`inject` samples non-followers — this is "recommended for you" and is what makes
cross-cluster spread possible. `k_inj = 0` reduces the system to pure
subscription, which is a legitimate and interesting condition.

**(b) Ranking.** Each user `u` has an inbox `I_u` of candidate posts. A ranking
function orders it:

```
rank: (u, I_u, X, D_p, state) → ordered list
```

| Ranker | Definition | Models |
|---|---|---|
| `chronological` | sort by `t` desc | classic timeline |
| `engagement_optimized` | sort by predicted `P(engage)` | modern feed |
| `affinity` | sort by topic/stance similarity | filter bubble maximal |
| `popularity` | sort by current engagement count | bandwagon feed |
| `random` | shuffle | null model |

**(c) Attention budget.** User sees only the top `B_u` items, with
position-dependent attention decay:

```
B_u ~ Poisson(b · activity_u)
P(see item at rank r) = exp(−r / τ_pos)
```

**The finite attention budget is load-bearing.** Without it, exposure grows
without bound, virality is unconstrained, and every post competes with nothing.
Attention is the scarce resource the whole system is fighting over; if it is not
scarce, none of the interesting dynamics appear.

### 2.6 Reaction — the engagement kernel

Formulated as a multinomial logit over actions, with `skip` as the reference
category (utility fixed at 0):

```
U_a(u, p) = θ_a^T φ(x_u, d_p, ctx)          a ∈ {like, reply, repost, quote, report}
U_skip    = 0
P(a | u, p) = exp(U_a) / (1 + Σ_{a'} exp(U_{a'}))
```

**The feature map `φ` is the theory of engagement.** Swapping theories means
swapping `φ` (and its `θ`), and nothing else in the codebase changes. This is
the central abstraction of the toolbox.

Feature vocabulary available to `φ`:

```
affinity      = ⟨a_u, topic_p⟩
agreement     = −||s_u − s_p||           (negative distance)
arousal       = arousal_p
arousal_x_neu = arousal_p · neuroticism_u
provoc_x_con  = provocativeness_p · contrarianism_u
prominence    = log(1 + followers(author_p))
social_proof  = log(1 + current_engagements_p)
tie_strength  = is_follower(u, author_p)
quality       = quality_p
novelty       = novelty_p
recency       = −(t − t_p)
credulity_x_q = credulity_u · (1 − specificity_p)
```

Named kernels to ship:

| Kernel | Dominant terms | Prediction |
|---|---|---|
| `homophily` | `affinity`, `agreement` | strong echo chambers, low conflict |
| `outrage` | `provoc_x_con`, `−agreement`, `arousal` | cross-cluster hostility, high volume |
| `bandwagon` | `social_proof`, `prominence` | extreme attention inequality, rich-get-richer |
| `epistemic` | `quality`, `novelty`, `specificity` | control: what if merit drove attention |
| `null` | intercept only | pure structural baseline |

The critical sign: for `outrage`, the coefficient on `agreement` is **negative**
for high-contrarianism users. Disagreement increases engagement. Without a term
of this shape the simulation contains no arguments.

### 2.7 Cascades

A `repost` or `quote` creates a derived post inheriting `root_p`, with dims
perturbed (quotes shift stance toward the quoter, reposts do not). It re-enters
exposure with the reposter as author, and a depth decay:

```
visibility multiplier = ρ^depth,  ρ ≈ 0.7
```

The cascade is a branching process with effective reproduction number

```
R = E[# reposts per exposure] · E[audience per repost]
```

Calibrate so `E[R] < 1` (cascades usually die) but `Var[R]` is large enough that
the tail crosses 1. Subcritical-with-heavy-tail is the regime that reproduces
observed cascade size distributions. Log `R_eff` per tick as a diagnostic; if it
sits above 1 the run will saturate and the results are meaningless.

Hard caps: max depth, max total cascade size. Trip a warning rather than
silently truncating.

### 2.8 Discourse state update

```
s(t+1)    = ρ_s · s(t) + (1 − ρ_s) · normalize(Σ_p w_p · topic_p)
σ(t+1)[k] = ρ_σ · σ(t)[k] + (1 − ρ_σ) · weighted_mean(stance_p : topic_p = k)
w_p       = engagement_count_p
```

Weighting by engagement rather than post count is what lets a small number of
highly-engaged users capture the agenda — a specific, testable claim about
discourse capture that the model can now be interrogated about.

### 2.9 Trait drift

Two channels plus mean reversion. Operates on stored (unconstrained) traits.

**Channel 1 — reinforcement (every tick, free).** Users learn what plays. A
bandit-style update on expression dims only:

```
r_p     = engagement_p − E[engagement | author_p]     surprise, not raw count
Δ^rl_u  = lr · plasticity_u · Σ_p r_p · (d_p^exp − x_u^exp)
```

Moves the author's expression style toward the style of their
better-than-expected posts. This alone generates measurable style convergence
under an engagement-optimized ranker, which is a headline result the toolbox
should be able to produce without any LLM involvement.

**Channel 2 — social influence (every tick, free).** Exposure-weighted pull of
stance toward what the user consumed and did not reject:

```
Δ^soc_u = lr_s · (1 − conviction_u) · Σ_{p ∈ consumed_u} weight(action) · (s_p − s_u)
weight: like +1, repost +1.5, reply −0.5, report −2, skip 0
```

**Channel 3 — LLM adjudication (rare, event-triggered).** Only on salient
events: a post in the top 1% of engagement, a pile-on (`>k` hostile replies
received), or sustained cross-cluster exposure. The LLM sees the voice card, a
digest of the event, and returns bounded deltas with justification. This is the
only place the LLM touches dynamics, and it is gated to keep both cost and
non-determinism bounded.

**Composition — Ornstein–Uhlenbeck:**

```
X ← X + plasticity ⊙ (Δ^rl + Δ^soc + clip(Δ^llm, −ε, ε))
      − k ⊙ (X − Bs)
      + σ_noise · ξ,   ξ ~ N(0, I)

Bs ← Bs + k_b · (X − Bs),    k_b ≈ k / 10
```

The mean-reversion term is not optional. Without it, additive deltas compound
and the population becomes uniformly extremal within a few hundred ticks;
this failure is monotone and quiet, so it is easy to mistake for a finding.
Slow baseline drift is what permits genuine long-term radicalization while
keeping short-term mood excursions recoverable.

Per-block reversion rates: expression reverts fast (`k` large — style is
fashion), stance reverts slowly, personality effectively not at all.

### 2.10 LLM realization (offline pass)

Runs over a completed `Run`. Never inside the tick.

**Voice cards.** One call per user, cached forever, keyed by
`hash(archetype, quantized_traits)` so similar users share cards and the cache
hit rate stays high:

```
input:  trait vector rendered as a labeled feature list
output: 3-sentence persona + 3 concrete writing tics + register notes
```

**Rendering.** Batched, 20–50 posts per call:

```
input:  voice_card, post dims (as labeled bands: "arousal: high"),
        thread context (parent chain, truncated), topic label
output: post text only
```

**Lazy realization.** Render only the subgraph a human will read. A run of 10⁶
interactions can be fully analyzed numerically; text is generated for the few
thousand posts you actually inspect. Realization is a view on the run, not part
of it.

**Traits → prompt.** Do not put raw floats in the prompt. Quantize each trait to
a 5-point band with a verbal label. Floats produce a model that ignores them;
labels produce one that acts on them.

---

## 3. Algorithmic pipeline

### 3.1 Tick loop

```
for t in range(T):
    # 1. Global state
    s, σ = decay_discourse(s, σ, ρ)
    circ  = circadian(t, phases)

    # 2. Generation
    n_posts     = rng.poisson(activity * circ * fatigue)
    authors     = repeat(arange(N), n_posts)
    topics      = sample_topics(affinity[authors], s, η)
    dims        = A @ X[authors].T + B @ s + C[topics] + noise
    replies     = hawkes_draw(open_threads, t)          # reply targets
    posts.append(authors, dims, topics, t, parents)

    # 3. Exposure
    inbox       = scatter_posts_to_followers(new_posts, G_csr, k_inject)
    ranked      = ranker(inbox, X, dims, state)          # SWAPPABLE
    seen        = topk_per_user(ranked, budget)          # (user, post) pairs

    # 4. Reaction
    feats       = φ(X[seen.user], dims[seen.post], ctx)  # SWAPPABLE
    utils       = feats @ Θ.T                            # (S, |A|)
    actions     = gumbel_argmax(utils, rng)              # incl. skip
    engagements.append(seen[actions != SKIP], actions)

    # 5. Cascades
    for wave in range(max_depth):
        reposts = engagements[actions == REPOST]
        if empty: break
        derive_posts(reposts, ρ_depth); goto 3 for these only

    # 6. Drift
    X, Bs = drift_step(X, Bs, engagements, seen, cfg)    # SWAPPABLE
    if t % llm_interval == 0:
        flag_salient_events(engagements)                 # queued, not executed

    # 7. Update discourse
    s, σ = update_discourse(s, σ, new_posts, engagements)
```

### 3.2 Vectorization strategy

The one thing that must not be a Python loop is exposure scatter. Pattern:

```python
# authors: (P,) post → author
# G_csr:   follower lists per author
counts   = G_csr.indptr[authors+1] - G_csr.indptr[authors]
post_ids = np.repeat(np.arange(P), counts)
user_ids = np.concatenate([G_csr.indices[G_csr.indptr[a]:G_csr.indptr[a+1]]
                           for a in authors])           # or ragged gather
```

The list comprehension above is the honest but slow version. Replace with a
precomputed ragged gather using `np.add.reduceat` / `awkward-array`, or cap
per-post fan-out by sampling `min(followers, cap)` followers for hub authors.
The cap is defensible modeling, not just a performance hack: nobody's post
actually reaches all ten million followers.

Top-k per user over a ragged `(user, post, score)` array: `np.lexsort` on
`(−score, user)` then `np.add.reduceat` for segment boundaries, then a mask on
within-segment rank.

### 3.3 Complexity

| Phase | Cost |
|---|---|
| Generation | `O(P · n)` |
| Exposure scatter | `O(Σ_p |followers(author_p)|)` ← dominant, capped |
| Ranking | `O(E log E)` from the lexsort |
| Reaction | `O(S · |φ|)`, one matmul |
| Drift | `O(N · n)` |

Target: `N = 10⁴`, `T = 500` in under 60s single-core. If a tick exceeds ~100ms,
the exposure scatter is the culprit.

### 3.4 RNG discipline

One root seed. Spawn independent child generators per phase:

```python
root = np.random.default_rng(seed)
rngs = {name: np.random.default_rng(s)
        for name, s in zip(PHASES, root.spawn(len(PHASES)))}
```

Phase-independent streams mean changing the number of posts in tick 3 does not
change which users the graph generator connected. Without this, parameter sweeps
are confounded by RNG stream misalignment and you cannot attribute differences
to the parameter.

### 3.5 What is and is not persisted

Persist: `X` snapshots (every `k` ticks), `Bs`, posts + dims, engagements, graph,
config, per-tick metrics.

Do not persist: exposures. At `N=10⁴` exposures outnumber engagements ~50:1 and
dominate everything. Retain per-tick exposure *counts* and a fixed random sample
(1%) for diagnostics.

---

## 4. Technical architecture

### 4.1 Module layout

```
discourse_lab/
  config.py         Config, KernelSpec, frozen dataclasses, hashing
  state.py          Population, Posts, Run containers (all SoA)
  population/
    marginals.py    marginal distributions registry
    copula.py       correlated sampling, Higham nearest-PSD
    archetypes.py   archetype definitions
  network/
    latent_space.py  sbm.py  configuration.py  barabasi.py
  dynamics/
    timing.py       circadian, Poisson, Hawkes
    generation.py   post dim generation, the A/B/C maps
    exposure.py     scatter, inject, budget
    rankers.py      chronological / engagement / affinity / popularity / random
    kernels.py      φ feature maps + Θ, the named theories
    cascade.py      branching, depth decay, R_eff diagnostics
    drift.py        RL, social, OU composition
  llm/
    voice.py        voice card generation + cache
    render.py       batched post realization
    adjudicate.py   gated drift proposals
  metrics/
    stylized.py     target distributions
    polarization.py  clustering.py  inequality.py
  experiment/
    runner.py       run(cfg, seed) → Run
    sweep.py        cartesian product, parallel, cached
    compare.py      Run × Run → diff report
  io/
    store.py        parquet/pickle, config-hash keyed
```

### 4.2 Component interfaces

Every swappable component is a `Protocol`. This is what makes it a toolbox
rather than one simulation.

```python
class GraphGenerator(Protocol):
    def __call__(self, pop: Population, cfg: Config, rng: Generator) -> sparse.csr_matrix: ...

class Ranker(Protocol):
    def __call__(self, inbox: Inbox, pop: Population, posts: Posts,
                 state: GlobalState, rng: Generator) -> np.ndarray: ...  # scores

class EngagementKernel(Protocol):
    feature_names: list[str]
    theta: np.ndarray                     # (|actions|, |features|)
    def features(self, users: np.ndarray, posts: np.ndarray,
                 pop: Population, ps: Posts, ctx: Context) -> np.ndarray: ...

class DriftModel(Protocol):
    def __call__(self, pop: Population, ev: Engagements,
                 cfg: Config, rng: Generator) -> np.ndarray: ...  # ΔX
```

Registered by name so configs stay serializable:

```python
REGISTRY: dict[str, EngagementKernel] = {}

@register("outrage")
class OutrageKernel: ...
```

Config then holds `kernel: str = "outrage"`, and a run is fully described by
JSON. Sweeps over *theories* become sweeps over strings.

### 4.3 Config

```python
@dataclass(frozen=True)
class Config:
    n_users: int = 10_000
    n_ticks: int = 500
    n_topics: int = 8
    stance_dims: int = 3

    graph: str = "latent_space"
    graph_params: FrozenDict = ...
    ranker: str = "chronological"
    kernel: str = "homophily"
    kernel_theta: FrozenDict = ...        # coefficient overrides
    drift: str = "full"

    attention_budget: float = 30.0
    inject_k: int = 0
    hawkes_alpha_beta: float = 0.6
    ou_k: FrozenDict = ...
    llm_adjudication: bool = False

    def hash(self) -> str:
        return blake2b(canonical_json(self), digest_size=8).hexdigest()
```

### 4.4 Run artifact and sweeps

```python
@dataclass
class Run:
    config: Config
    seed: int
    pop_snapshots: dict[int, np.ndarray]
    posts: PostTable
    engagements: np.ndarray
    graph: sparse.csr_matrix
    tick_metrics: pl.DataFrame

def run(cfg: Config, seed: int) -> Run          # pure
def cached_run(cfg, seed) -> Run                # keyed on (cfg.hash(), seed)

def sweep(base: Config, grid: dict[str, list], seeds: list[int]) -> pl.DataFrame:
    """Returns tidy long-form: one row per (config, seed, metric)."""
```

**Always sweep over multiple seeds.** Cascade dynamics give enormous run-to-run
variance; a single run per condition tells you essentially nothing. Minimum 10
seeds, and report distributions rather than points. This is the single most
common way simulation studies of this kind go wrong.

### 4.5 Notebook ergonomics

```python
runs = sweep(base, {"kernel": ["homophily", "outrage", "bandwagon", "null"]},
             seeds=range(20))
plot_metric(runs, "cross_cluster_hostility", by="kernel")
```

No database. Parquet in `./runs/{cfg_hash}/{seed}.parquet`. A 10k-user, 500-tick
run is a few hundred MB uncompressed and considerably less on disk.

---

## 5. Metrics and validation

### 5.1 Stylized facts to reproduce

The simulation is calibrated when these emerge without being imposed:

| Fact | Target |
|---|---|
| Engagement per post | heavy-tailed, median ≈ 0, α ∈ [2, 3] |
| Cascade size | power law, >90% of cascades size 1 |
| Thread depth | approximately geometric, mean 1.5–3 |
| Attention Gini | 0.8–0.95 |
| Posting volume Gini | 0.7–0.9 |
| Reciprocity | 0.2–0.4 |
| Clustering coefficient | ≫ random graph of same degree |
| Inter-cluster interaction rate | low, but hostility rate high given contact |

### 5.2 Experimental metrics

- **Polarization**: bimodality coefficient of stance projections; distance
  between cluster stance centroids over time.
- **Echo chamber index**: fraction of a user's consumed stance mass within `δ`
  of their own.
- **Attention inequality**: Gini, top-1% share.
- **Quality–attention correlation**: Spearman(`quality_p`, engagement). Under
  `bandwagon` this should be near zero — verify it is.
- **Style convergence**: trace of expression-block covariance over time.
- **Drift magnitude**: `||X_t − X_0||` by trait block, by archetype.

### 5.3 Null comparison protocol

Every reported effect is a difference against a matched null: same population,
same graph, same activity, `kernel="null"`. If an effect survives against the
null, it is attributable to the engagement theory rather than to the structural
priors. If it does not — and many will not — that is itself the finding.

### 5.4 Sensitivity

Before drawing conclusions, one-at-a-time sensitivity over: `attention_budget`,
`inject_k`, `hawkes_alpha_beta`, `ou_k`, mean degree. Any conclusion that
inverts within a plausible range of one of these is a conclusion about that
parameter, not about the mechanism under study.

---

## 6. Implementation order

1. **Skeleton + config + RNG discipline.** Empty tick loop that runs and caches.
2. **L0 population + copula.** Validate marginals and correlations against spec.
3. **L1 graph.** Validate degree distribution, clustering, homophily.
4. **L2 generation** with a stub renderer emitting `"[u17 · topic3 · hostile]"`.
5. **L3 exposure + reaction**, `homophily` kernel only.
6. **Metrics + stylized fact validation.** Calibrate here. Do not proceed until
   the distributions in §5.1 are right.
7. **Cascades**, verify `R_eff` diagnostics.
8. **Additional kernels and rankers.** First real experiments happen here, still
   with zero LLM calls.
9. **Drift**, channels 1 and 2. Verify no runaway over 1000 ticks.
10. **LLM realization pass** (voice cards, rendering).
11. **LLM adjudication**, gated, last.

Steps 1–9 have no LLM dependency at all. If the numeric dynamics do not produce
plausible distributions, adding language will conceal the problem rather than
fix it.

---

## 7. Open modeling choices

Decisions deliberately left to the implementer, each with consequences:

1. **Topic representation**: one-hot (simple, interpretable) vs simplex
   (multi-topic posts, richer but harder to read). Recommend one-hot first.
2. **Graph static or dynamic?** Follow/unfollow in response to engagement adds a
   powerful feedback loop (self-sorting into echo chambers) but greatly
   complicates attribution. Recommend static for v1, dynamic as an explicit
   experimental condition later.
3. **Do users leave?** Churn as a function of received negative engagement makes
   the population endogenous and can produce survivorship effects that look like
   radicalization. Interesting, but confounds everything before it.
4. **Is `quality` observable to users?** If exposure to quality is noisy,
   `epistemic` kernels degrade gracefully. If perfect, that kernel is unrealistic.
5. **Multiple stance dims — orthogonal or correlated?** Correlated axes produce
   the empirically observed collapse to a single dominant dimension. Consider
   making this an experimental condition rather than an assumption.
6. **Bot / coordinated-inauthentic population?** A small archetype with
   scripted, non-drifting behavior and inflated activity. Cheap to add, and one
   of the more useful things this toolbox can be pointed at.
