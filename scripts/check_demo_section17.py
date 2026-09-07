# Executes ONLY the appended section-17 code cells, with the notebook's
# earlier namespace reconstructed. Full-notebook execution belongs to the
# user's kernel (LLM/intervention sections are guarded there).
import dataclasses
import os
import sys
import tempfile

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DLAB_HOME"] = tempfile.mkdtemp()

import json

from discourse_lab.config import Config
from discourse_lab.data import scenario_config
from discourse_lab.dynamics.posts import concat_post_batches
from discourse_lab.io.workspace import workspace  # noqa: F401
from discourse_lab.population import cached_population
from discourse_lab.network import cached_graph
from discourse_lab.runner import cached_run, load_run, phase_rngs, run_iter
from discourse_lab.semantics import lexicon_for

SEED = 0
rng = np.random.default_rng(SEED)

cfg = scenario_config(dataclasses.replace(
    Config(),
    population=dataclasses.replace(Config().population, n_users=600, n_topics=8),
    dynamics=dataclasses.replace(Config().dynamics, n_ticks=30, kernel="homophily",
                                 ranker="affinity", drift="none"),
))
pop = cached_population(cfg, SEED, phase_rngs(SEED)["population"])
lex = lexicon_for(cfg)

with open("notebooks/demo.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

ns = {
    "dataclasses": dataclasses, "np": np, "pl": pl, "os": os,
    "cfg": cfg, "pop": pop, "lex": lex, "SEED": SEED, "rng": rng,
    "cached_population": cached_population, "cached_graph": cached_graph,
    "run_iter": run_iter, "cached_run": cached_run, "load_run": load_run,
    "phase_rngs": phase_rngs, "concat_post_batches": concat_post_batches,
}

ran = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code" or i < 47:
        continue
    src = "".join(cell["source"])
    print(f"--- executing cell {i} ---")
    exec(compile(src, f"<cell {i}>", "exec"), ns)
    ran += 1

print(f"\n{ran} new code cells executed cleanly")
