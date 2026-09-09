"""Experiment 01 -- Sorting vs. Selective Exposure.

Two theories of why online discourse fragments:

  World A ("the algorithm sorts you")  -- an outrage-tuned engagement kernel,
      an engagement-optimizing ranker, position-only selection (users take
      whatever the feed hands them), heavy algorithmic injection (inject_k=20)
      and slow rewiring on.
  World B ("you sort yourself")        -- a homophily kernel, a plain
      chronological ranker (no algorithmic sorting at all), homophilous
      selection (users actively pick agreeable content out of an unsorted
      feed), light injection (inject_k=2) and no rewiring.

Each is run against its own matched `kernel="null"` twin (spec's mandatory
comparison protocol) -- same population, same graph, same activity, only the
kernel swapped for a content-blind one -- so every effect reported is a
model-minus-null delta, never a raw level.

World C (spec S10, "digital micropublics" / civic kernel + SBM blocks seeded
on topic_affinity) is NOT implemented here: the machinery it needs (a "civic"
kernel, sbm_blocks keyed on topic affinity) does not exist in this codebase
and the spec itself flags C as a follow-up reconstruction, not a required
run. See the notebook's "Deviations from the spec" section for the full list
of what was scaled down or skipped.

This script:
  1. runs a smoke test (1 seed, short) of every cell to sanity-check the API
     and estimate wall-clock cost,
  2. runs the C9 stylized gate on the shared substrate (prerequisite 1),
  3. runs the main 2x2 grid (World A / World B x model / null) across N_SEEDS
     seeds,
  4. runs the inject_k ladder [0, 2, 5, 10, 20] for both worlds at a reduced
     seed count, if OPTIONAL steps are enabled,
  5. writes tidy summary CSVs to results/experiment01/.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

import discourse_lab.exposure  # noqa: F401  (registers kernels/rankers)
from discourse_lab.analysis import set_param
from discourse_lab.config import Config
from discourse_lab.metrics import (
    affective_distance,
    animus_asymmetry,
    cluster_centroid_distance,
    ident_animus_coupling,
)
from discourse_lab.metrics.stylized import stance_clusters
from discourse_lab.outcomes import normative_outcomes
from discourse_lab.population import cached_population
from discourse_lab.population.links import to_used
from discourse_lab.runner import cached_run, load_run, phase_rngs
from discourse_lab.semantics import Lexicon

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "experiment01"

# --------------------------------------------------------------------------
# scale-down (documented deviation from the spec's 10,000 users / 500 ticks /
# 20 seeds full run plan -- see the notebook for the full rationale)
# --------------------------------------------------------------------------
N_USERS = 1500
N_TICKS = 150
MAIN_SEEDS = tuple(range(8))          # 8 seeds for the main A/B x null grid
LADDER_SEEDS = tuple(range(4))        # 4 seeds for the inject_k ladder
GATE_SEEDS = (0, 1)                   # 2 seeds for the pre-flight gate
PERSIST = ("posts", "exposures", "engagements", "traits")

TRACKED_OUTCOME_KEYS = (
    "cross_cutting_exposure.stance_share",
    "cross_cutting_exposure.camp_share",
    "voice_inequality.attention_gini",
    "voice_inequality.top1_share",
    "epistemic_alignment.quality_attention_rho",
    "feed_narrowing.bubble",
    "hostility_given_contact.contact_rate",
    "hostility_given_contact.hostility",
)


def _bimodal_axis() -> dict:
    """A one-axis, two-camp stance scenario (mirrors notebooks/demo.ipynb's
    C1 cell). Both theories are about camp-conditional affect dynamics
    (`ident_animus_coupling`, `affective_distance`, `animus_asymmetry` all
    gate on `camps_and_bimodality`), and the packaged default scenario's
    stance marginal measures bimodality 0.297 against the 5/9 gate --
    camps never form, so the affect channel never activates and every
    kernel/world looks identical on those metrics regardless of what the
    kernel actually does. This axis measures 0.75, comfortably over the
    gate.
    """
    n_bins = 128
    xs = -1 + 2 * (np.arange(n_bins) + 0.5) / n_bins
    dens = np.exp(-((xs + 0.6) ** 2) / (2 * 0.2**2)) + np.exp(-((xs - 0.6) ** 2) / (2 * 0.2**2))
    return {
        "name": "camp", "pole_neg": "camp A", "pole_pos": "camp B",
        "marginal": {"kind": "empirical", "bins": n_bins, "support": [-1, 1],
                     "density": (dens / dens.sum() * n_bins).tolist()},
        "expression_cost": {"neg": 0.0, "pos": 0.0},
    }


def base_config() -> Config:
    """The shared substrate both worlds and both nulls are built from."""
    from discourse_lab.config import ScenarioConfig

    cfg = Config(
        population=dataclasses.replace(
            Config().population, n_users=N_USERS, affect=True, stance_dims=1),
        scenario=ScenarioConfig(stance_axes=(_bimodal_axis(),)),
    )
    cfg = set_param(cfg, "dynamics.n_ticks", N_TICKS)
    cfg = set_param(cfg, "dynamics.exposure_sample_rate", 0.05)
    return cfg


def world_a(base: Config) -> Config:
    cfg = set_param(base, "dynamics.kernel", "outrage")
    cfg = set_param(cfg, "dynamics.ranker", "engagement_optimized")
    cfg = set_param(cfg, "dynamics.selection", "position_only")
    cfg = set_param(cfg, "dynamics.inject_k", 20)
    cfg = set_param(cfg, "dynamics.rewire", True)
    return dataclasses.replace(cfg, label="exp01-world_a")


def world_b(base: Config) -> Config:
    cfg = set_param(base, "dynamics.kernel", "homophily")
    cfg = set_param(cfg, "dynamics.ranker", "chronological")
    cfg = set_param(cfg, "dynamics.selection", "homophilous")
    cfg = set_param(cfg, "dynamics.inject_k", 2)
    cfg = set_param(cfg, "dynamics.rewire", False)
    return dataclasses.replace(cfg, label="exp01-world_b")


def null_twin(cfg: Config, label: str) -> Config:
    """Matched null: kernel -> "null", everything else held fixed."""
    return dataclasses.replace(
        set_param(cfg, "dynamics.kernel", "null"), label=label
    )


CELLS = {
    "world_a": world_a,
    "world_b": world_b,
}


# --------------------------------------------------------------------------
# affect-derived metrics (need the population's final animus/identification,
# which live in traits.parquet in "stored" space and must be relinked)
# --------------------------------------------------------------------------

def _affect_metrics(handle, cfg: Config, seed: int) -> dict:
    if not cfg.population.affect or not handle.has_traits:
        return {
            "affective_distance": float("nan"),
            "animus_asymmetry": float("nan"),
            "ident_animus_coupling": float("nan"),
            "stance_centroid_distance": float("nan"),
        }
    rngs = phase_rngs(seed)
    pop = cached_population(cfg, seed, rngs["population"])
    lex = Lexicon.from_handle(handle)

    traits = handle.traits()
    t_max = int(traits["t"].max())
    last = traits.filter(pl.col("t") == t_max).sort("user")
    idx = {name: i for i, name in enumerate(pop.trait_names)}

    def col(name: str) -> np.ndarray:
        stored = last[f"x_{idx[name]}"].to_numpy()
        link = pop.links[idx[name]]
        return to_used(stored, link)

    identification = col("identification")
    animus = col("animus")

    stance_cols = lex.stance_columns(pop.trait_names)
    stance_final = np.column_stack([col(pop.trait_names[i]) for i in stance_cols])
    labels = stance_clusters(stance_final)

    return {
        "affective_distance": float(affective_distance(animus, labels)),
        "animus_asymmetry": float(animus_asymmetry(animus, labels)),
        "ident_animus_coupling": float(ident_animus_coupling(identification, animus)),
        "stance_centroid_distance": float(cluster_centroid_distance(stance_final, labels)),
    }


def run_cell(cfg: Config, seed: int) -> dict:
    cached_run(cfg, seed, persist=PERSIST)
    handle = load_run(cfg, seed)
    rngs = phase_rngs(seed)
    pop = cached_population(cfg, seed, rngs["population"])
    lex = Lexicon.from_handle(handle)

    row: dict = {"label": cfg.label, "seed": seed}
    row.update(normative_outcomes(handle, pop, lex))
    row.update(_affect_metrics(handle, cfg, seed))
    return row


def run_grid(cells: dict[str, Config], seeds) -> pl.DataFrame:
    rows = []
    for label, cfg in cells.items():
        for seed in seeds:
            t0 = time.time()
            row = run_cell(cfg, seed)
            row["wall_s"] = time.time() - t0
            rows.append(row)
            print(f"  {label} seed={seed}: {row['wall_s']:.1f}s")
    return pl.DataFrame(rows)


def smoke_test() -> None:
    base = set_param(base_config(), "dynamics.n_ticks", 10)
    cfgs = {"world_a": world_a(base), "world_b": world_b(base)}
    cfgs["null_a"] = null_twin(cfgs["world_a"], "exp01-null_a-smoke")
    cfgs["null_b"] = null_twin(cfgs["world_b"], "exp01-null_b-smoke")
    for name, cfg in cfgs.items():
        t0 = time.time()
        run_cell(cfg, seed=0)
        print(f"smoke {name}: {time.time() - t0:.1f}s for 10 ticks, n_users={cfg.population.n_users}")


def run_gate() -> None:
    from discourse_lab.experiments.gate import stylized_gate

    base = base_config()
    # gate against the "null"-kernel, chronological-ranker shared substrate
    gate_cfg = set_param(base, "dynamics.kernel", "null")
    gate_cfg = set_param(gate_cfg, "dynamics.ranker", "chronological")
    report = stylized_gate(gate_cfg, seeds=GATE_SEEDS, n_ticks=min(60, N_TICKS))
    print(report.summary())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gate_report.json").write_text(
        json.dumps({"passed": report.passed, "failures": report.failures, "rows": report.rows}, indent=2)
    )


def run_main_grid() -> pl.DataFrame:
    base = base_config()
    cfgs = {
        "world_a": world_a(base),
        "world_b": world_b(base),
    }
    cfgs["null_a"] = null_twin(cfgs["world_a"], "exp01-null_a")
    cfgs["null_b"] = null_twin(cfgs["world_b"], "exp01-null_b")

    df = run_grid(cfgs, MAIN_SEEDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(RESULTS_DIR / "main_grid.csv")
    return df


def run_inject_ladder() -> pl.DataFrame:
    base = base_config()
    cfgs = {}
    for k in (0, 2, 5, 10, 20):
        cfgs[f"world_a-k{k}"] = set_param(world_a(base), "dynamics.inject_k", k)
        cfgs[f"world_b-k{k}"] = set_param(world_b(base), "dynamics.inject_k", k)
    df = run_grid(cfgs, LADDER_SEEDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(RESULTS_DIR / "inject_ladder.csv")
    return df


if __name__ == "__main__":
    import sys

    steps = sys.argv[1:] or ["smoke", "gate", "main", "ladder"]
    if "smoke" in steps:
        smoke_test()
    if "gate" in steps:
        run_gate()
    if "main" in steps:
        run_main_grid()
    if "ladder" in steps:
        run_inject_ladder()
