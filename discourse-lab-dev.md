# Discourse Lab — Development Notes

Companion to `discourse-lab-design-v2.md`, which holds the model. This document
covers packaging, environment, notebook integration, and build order. Where the
two disagree, the design document wins on modelling and this one wins on
mechanics.

Status markers: **[settled]** decided, **[open]** still to decide.

---

## 1. Target

**[settled] `N` between 10,000 and 20,000.** The scenario is a closed-off area
of society, not a whole platform.

Consequences of that size:

- Everything fits in memory. 20k × 40 traits × 500 ticks ≈ 1.6 GB in float64,
  half that in float32. **Snapshot every tick.** Drift trajectories become
  post-hoc analysis rather than something that must be decided before the run.
- Frontier expansion is not a performance problem. A 3-hop neighbourhood in a
  20k graph is on the order of thousands of nodes — enumerable. Frontier
  sampling (§7.1 of the design) stays as a *modelling* choice about realistic
  reach, not a performance necessity.
- A tick should be single-digit milliseconds. If it is not, the cause is
  algorithmic, not the absence of a JIT.

**[settled] Small-`N` modelling caveat.** A 20k community with mean degree ~50
has a small diameter; information crosses it quickly and there is limited room
for separate epistemic regions to persist. For the §4.2 bubble dynamics to have
somewhere to live, local neighbourhoods must actually differ — which argues for
SBM or strongly homophilous latent-space as the **default** graph generator, not
Barabási–Albert.

---

## 2. Environment

**[settled] Must run unchanged in a local Jupyter and in Colab.**

That constraint implies three things, and they drive most of what follows: no
build step at install time, no absolute paths, installable from a git URL.

### 2.1 Dependencies

```
numpy          state, all vectorised math
scipy.sparse   graph (CSR + CSC)
polars         per-tick metrics tables
pyarrow        parquet persistence
anywidget      notebook widgets
```

**[settled] No numba, no torch.** At 20k neither earns its place, and numba is
the most likely dependency to break a fresh Colab install.

### 2.2 Paths

**[settled] One resolver. Nothing hardcoded.**

```python
def workspace() -> Path:
    if p := os.environ.get("DLAB_HOME"):
        return Path(p)
    if Path("/content").exists():          # Colab
        return Path("/content/dlab")
    return Path.cwd() / "dlab"
```

`runs/` and `scenarios/` live under it. Colab sessions are ephemeral; document
mounting Drive and setting `DLAB_HOME` for runs that must survive, but do not
build Drive integration in — it is a one-line user action.

**[settled] Ship default scenarios as package data.** A fresh Colab must be able
to run something with no files present:

```python
from discourse_lab import demo
run = demo.quick()
```

Otherwise the first experience of the toolbox is a missing-file error.

### 2.3 Reproducibility

**[settled] Target statistical reproducibility across environments, not bitwise.**

`np.random.default_rng` (PCG64) is stable across OS and numpy versions, so the
random streams themselves are fine. The risks are elsewhere:

- BLAS thread count changes reduction order in matmuls.
- Dict or set iteration in the op registry would make op order
  machine-dependent. Additive composition means order is mathematically
  irrelevant, but floating-point addition is not associative, so results can
  differ in the last bits.

*Rules that follow:* keep the op registry a **sorted list**, never a dict
iteration order; never branch on exact float equality anywhere in the tick.

Bitwise reproducibility across Colab and local is achievable but fragile.
Statistical reproducibility across 20 seeds is robust and is what the work
actually requires. Promise the second.

---

## 3. Package layout

**[settled] A package, installed. The notebook holds no logic.**

```
discourse_lab/
  __init__.py
  config.py          nested frozen configs (§8.2): Population / Graph /
                     Dynamics / Scenario, plus WorldConfig
  state.py           Population, Posts, GlobalState, Run — struct-of-arrays
  registry.py        phase and component registration; sorted, deterministic
  runner.py          run(), run_iter()
  sweep.py           cartesian product over configs × seeds
  population/        marginals, copula, archetypes, empirical inverse CDF
  network/           latent_space, sbm, configuration, barabasi
  dynamics/
    timing.py        circadian, Poisson, Hawkes
    generation.py    post dimension generation
    exposure.py      candidate generators, ranking, attention budget
    kernels.py       feature maps φ and coefficients Θ
    cascade.py       frontier advance
    perception.py    F_local, F_global, w_u blending
    drift.py         all upkeep ops
  measures/          registered F components
  metrics/           analysis over a completed Run
  widgets/
    __init__.py
    _static/         COMMITTED built JS — see §4.1
    stance_editor.py
    run_monitor.py
  data/
    scenarios/       shipped defaults
  demo.py

notebooks/           thin: configure, run, plot
dlab/                workspace — runs/, scenarios/ (gitignored)
```

Installed with `pip install -e .` locally, `pip install git+https://…` in Colab.

### 3.1 Registry must fire at import

Registration decorators run at module load. **This is why the notebook holds no
logic:** hot-reloading decorated functions defined in cells produces duplicate
registrations and stale closures. `%autoreload 2` handles the edit-run cycle for
package code correctly.

### 3.2 Accessors over raw indices

**[settled] Thin named views onto the state matrix.**

`X[:, 14]` is fast and unreadable, and mis-indexing is a silent failure.

```python
pop.contrarianism        # returns a view, not a copy
pop.stance               # (N, D) view
```

Zero runtime cost, removes an entire class of bug.

---

## 4. Notebook integration

### 4.1 anywidget

**[settled] Commit the built JS.** anywidget bundles its JS as package data.
Colab cannot run esbuild at install time, so the bundle must already exist in
the wheel. Build locally with esbuild in watch mode during widget development;
commit the output to `widgets/_static/`. Ugly in a repo, and it is what makes
`pip install git+https://…` work in a fresh Colab cell.

**[settled] Widget state is backed by files, not by the kernel.**

The stance editor autosaves to `scenarios/<name>.json` on every change. The
widget is then a *view onto a file* rather than live kernel state. This solves
three problems at once:

- Re-running a cell does not lose the drawing.
- The file-export path and the widget path become the same mechanism, not two.
- Colab session death loses nothing that was not already lost.

**[settled] Two widgets, in this order:**

1. **Stance editor** — already built as React (`stance-editor.jsx`); port to
   anywidget. Draw density curves per stance axis, live sampler preview, emits
   scenario config. Tight loop, exists already.
2. **Run monitor** — live plot of agreement measures, bubble index and attention
   Gini *while a run executes*. Tells you within twenty ticks whether a config
   will do anything, instead of waiting for completion. This is the difference
   between running a sweep blind and steering one.

### 4.2 Purity versus interactivity

**[settled] Generator core, pure wrapper.**

`run(cfg, seed) -> Run` is the right contract, but interactively you want to
inspect tick 200 without rerunning from zero.

```python
def run_iter(cfg, seed) -> Iterator[State]: ...   # break, inspect, resume
def run(cfg, seed) -> Run:                        # collects run_iter
```

Costs nothing and makes debugging viable. The run monitor consumes `run_iter`
directly.

---

## 5. Persistence

**[settled] Parquet, keyed by config hash.** No database.

```
dlab/runs/{cfg_hash}/{seed}/
    meta.json          full config, serialised
    traits.parquet     per-tick snapshots, one row group per tick (§7.3)
    posts.parquet
    engagements.parquet
    metrics.parquet    per-tick measures
    graph.npz          scipy sparse
```

`cached_run(cfg, seed)` checks the directory before executing. Sweeps are
resumable by construction.

**[settled] Exposures are not persisted.** They outnumber engagements by roughly
50:1. Retain per-tick counts and a 1% sample for diagnostics.

**[settled] Config serialisation must be canonical** — sorted keys, fixed float
formatting — or the hash is unstable across sessions and the cache silently
misses.

---

## 6. Build order

Each step ends somewhere runnable. Nothing after step 1 requires an LLM.

1. **Skeleton.** Nested config and structural hashing (§8.2), workspace
   resolver, registry, `run_iter` with an empty tick, streaming parquet writer
   (§7.3), artifact cache. Verify: a run caches and reloads; changing a
   population field invalidates the population artifact and nothing else.
2. **Population.** Copula, archetypes, empirical inverse CDF from the editor's
   density arrays. Verify: marginals and rank correlations match the spec;
   drawn scenario curves reproduce in the sample.
3. **Stance editor as anywidget.** Port the React component. Verify: draw →
   autosave → load in Python → sample → histogram matches the curve.
4. **Graph.** SBM and latent-space first. Verify: degree distribution,
   clustering coefficient, homophily; neighbourhoods measurably differ (§1).
5. **Timing and generation**, stub renderer emitting `[u17 · topic3 · strong]`.
6. **Exposure and reaction.** One candidate generator, one ranker, one kernel.
   First runnable dynamics.
7. **Perception.** `F_local`, `F_global`, `w_u` blend, measure registry
   including the salience/stance agreement pair.
8. **Frontier cascades.** Verify: cascade size distribution heavy-tailed,
   boundary-crossing diagnostic works.
9. **Run monitor widget** and the post-run analysis module. Everything after
   this is faster to develop.
10. **Experiment 1** (design §5.2) — sweep the familiarity/virality blend.
11. **Drift ops**, gains ramped from zero one loop at a time. Verify: no runaway
    over 1000 ticks with all five live.
12. **LLM realization pass** — world config, band quantization, voice cards.

**[settled] Steps 1–11 have no LLM dependency.** If the numeric dynamics do not
produce plausible distributions, adding language conceals the problem rather
than fixing it.

---

## 7. Analysis, memory and parallelism

### 7.1 Run monitor scope

**[settled] Live metrics during the run, full analysis afterward.**

The monitor consumes `run_iter` and plots measures as ticks complete: the
salience/stance agreement pair, bubble index, attention Gini, cascade activity,
per-op drift contribution norms (§3 of the design — the cancellation
diagnostic).

Post-run analysis is a separate module over a completed `Run`: trajectories,
distributions, boundary-crossing traces, cross-seed comparison.

**Rendered feed samples stay out of the monitor.** Realization is an offline
pass over a completed run (design §10); pulling it into the live loop would
contradict that and put an API call inside the tick. Render on demand after the
run, from the saved artifact.

### 7.2 Snapshot dtype

**[settled] float64 throughout, including snapshots.**

Drift deltas are small and accumulate over hundreds of ticks; float32 storage
would discard precision that the drift trajectory analysis depends on. Revisit
only if memory becomes binding.

### 7.3 Memory: snapshots must stream

**[settled] Snapshots are appended to parquet incrementally, never accumulated
in memory.**

float64 snapshots every tick at `N = 20,000` and ~40 traits is roughly 1.6 GB
per run. Holding that in memory is fine for one run and fatal for eight parallel
ones — 13 GB before counting posts, engagements or the graph.

```python
writer = pq.ParquetWriter(path, schema)
for state in run_iter(cfg, seed):
    writer.write_batch(snapshot_batch(state))    # row group per tick
```

Memory per worker then stays flat regardless of run length. `Run` becomes a
lazy handle over the parquet files rather than an in-memory object; analysis
reads the columns and tick ranges it needs.

This makes §7.2 and §7.4 compatible. Without it they conflict directly.

### 7.4 Parallel sweeps

**[settled] Parallel by default, with runtime backend detection.**

`multiprocessing` has always been stdlib; what is new in 3.13 is the
free-threaded build (PEP 703), which is opt-in, a separate interpreter binary,
and not what a hosted notebook runs. Subinterpreter pools arrived later still.
So the backend is **detected at runtime rather than assumed**:

```python
def executor(n_workers):
    if free_threaded():          # sys._is_gil_enabled() is False
        return ThreadPoolExecutor(n_workers)
    if interpreter_pool_available():
        return InterpreterPoolExecutor(n_workers)
    if fork_or_spawn_usable():
        return ProcessPoolExecutor(n_workers)
    return SequentialExecutor()
```

`DLAB_WORKERS=1` forces sequential — necessary for debugging, since worker
tracebacks are much worse than in-process ones.

**[settled] Workers return paths, not Runs.** A worker executes
`run(cfg, seed)`, streams its output to parquet, and returns the directory path.
Pickling a multi-gigabyte `Run` back through a pipe costs more than recomputing
it.

**[settled] The cache is the coordination mechanism.** Each worker checks
`cached_run` before executing, so an interrupted sweep resumes by re-invocation
and duplicate work is skipped without any locking.

*Colab caveat:* process pools there are unreliable and worker crashes are
frequently silent. The detection chain must degrade to sequential rather than
hanging, and the sweep must report which cells completed rather than assuming
all did.

---

## 8. Sweep scheduling and artifact reuse

### 8.1 Granularity

**[settled] Flat work queue over `(config, seed)` pairs, ordered seed-major.**

Parallelising over configs with seeds sequential inside was the alternative, on
the grounds that it yields cleaner partial results. It does not: the cache
already records per-cell completeness, so partial results are queryable
regardless of scheduling, and analysis must handle ragged seed counts anyway —
report `n` per cell rather than assuming uniformity.

A flat queue also self-balances. Run durations vary substantially because
cascade dynamics vary; grouping by config would leave workers idle behind a slow
cell.

**Ordering matters more than granularity.** Emit seed 1 of every config, then
seed 2 of every config, and so on:

```python
for s in seeds:
    for cfg in configs:
        yield (cfg, s)
```

An interrupted sweep then leaves one seed across the whole parameter space —
a noisy but complete picture — rather than twenty seeds of the first three
configs and nothing else. For a sweep like Experiment 1, where the question is
the *shape* of a curve across the blend parameter, the early preliminary read is
worth much more than depth at one point.

### 8.2 Population and graph reuse

**[settled] Cache them as content-addressed artifacts, not shared between
cells.**

Regenerating an identical population and graph for every cell of a sweep is
wasteful, but sharing objects across cells couples runs that should be
independent. Memoising a pure function does neither:

```
dlab/artifacts/pop/{pop_hash}/{seed}.npz
dlab/artifacts/graph/{pop_hash}-{graph_hash}/{seed}.npz
```

Each cell requests its population by hash and gets it from disk if present. No
coupling, no shared mutable state, and the cache is correct by construction
because the key is derived from exactly the config that determines the artifact.

Storage is trivial at this scale: a 20k population at 40 traits is ~6 MB, and a
graph with mean degree 50 is ~1M edges, around 12 MB in CSR.

**[settled] Consequence — `Config` becomes nested.**

The sub-hash must be *structural*, not a hand-picked list of relevant fields. A
hand-maintained list will eventually omit a field, and the failure mode is
silent reuse of a stale artifact — the worst kind of bug in a system whose whole
output is statistical.

```python
@dataclass(frozen=True)
class Config:
    population: PopulationConfig    # hashes to pop_hash
    graph:      GraphConfig         # hashes to graph_hash
    dynamics:   DynamicsConfig
    scenario:   ScenarioConfig      # from the stance editor
```

Each sub-config hashes independently; the run hash is the hash of the whole. The
nesting is worth doing for its own sake — it also makes sweep grids readable,
since a grid usually varies one sub-config while holding the others fixed.
