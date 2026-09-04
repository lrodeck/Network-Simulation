"""Nested frozen configs with structural hashing (dev notes §5, §8.2).

Every sub-config hashes independently and canonically (sorted keys, fixed float
formatting), so artifact keys are derived from exactly the config that
determines the artifact. A run is fully described by JSON.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# canonical serialisation
# --------------------------------------------------------------------------

def _canonical(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("config contains non-finite float")
        return round(obj, 12)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 12)
    if isinstance(obj, np.ndarray):
        return [_canonical(x) for x in obj.tolist()]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _canonical(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if not f.name.startswith("_")
        }
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    raise TypeError(f"cannot canonicalise {type(obj)}")


def canonical_json(obj: Any) -> str:
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"))


def structural_hash(obj: Any) -> str:
    return hashlib.blake2b(canonical_json(obj).encode("utf-8"), digest_size=16).hexdigest()


class Hashable:
    """Mixin giving every sub-config an independent structural hash."""

    def hash(self) -> str:
        return structural_hash(self)

    def to_dict(self) -> dict:
        return _canonical(self)

    def to_json(self) -> str:
        return canonical_json(self)


# --------------------------------------------------------------------------
# sub-configs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PopulationConfig(Hashable):
    n_users: int = 10_000
    n_topics: int = 8
    stance_dims: int = -1                     # -1 → derived from scenario axes
    archetype_weights: tuple[tuple[str, float], ...] = ()   # () → library defaults
    archetype_offsets: tuple[tuple[str, str, float], ...] = ()  # (archetype, trait, offset)
    correlation_pairs: tuple[tuple[str, str, float], ...] = ()  # () → library defaults
    activity_sigma: float = 1.2
    pareto_alpha: float = 2.3
    topic_logit_sigma: float = 1.0


@dataclass(frozen=True)
class GraphConfig(Hashable):
    generator: str = "latent_space"           # latent_space | sbm | configuration | barabasi
    mean_degree: float = 40.0
    homophily_beta: float = 0.35              # β on latent distance
    prominence_gamma: float = 0.6             # γ on log(1 + prominence)
    reciprocity: float = 0.2
    fanout_cap: int = 400                     # max followers reached per post per tick
    knn_k: int = 150                          # candidate pool when N is large
    long_tie_fraction: float = 0.1            # uniform random component in kNN graphs
    sbm_blocks: int = 0                       # 0 → one block per archetype
    sbm_homophily: float = 0.8


@dataclass(frozen=True)
class DynamicsConfig(Hashable):
    n_ticks: int = 500
    posts_per_tick_rate: float = 0.02         # Poisson rate at activity = 1
    ticks_per_day: int = 24
    fatigue_decay: float = 0.9

    attention_budget: float = 30.0            # b in B_u ~ Poisson(b · activity)
    tau_position: float = 6.0                 # position decay exp(-r / tau)
    inject_k: int = 0                         # algorithmic injections per post
    ranker: str = "chronological"
    kernel: str = "homophily"
    kernel_theta: tuple[tuple[str, str, float], ...] = ()   # (feature, action, value)

    hawkes_mu0: float = 0.004                 # baseline reply intensity per tick
    hawkes_ratio: float = 0.6                 # alpha/beta, must stay < 1
    hawkes_beta: float = 1.5
    max_thread_age: int = 15                  # ticks a thread stays open for Hawkes

    trend_eta: float = 0.3                    # topic susceptibility to discourse state
    post_lifetime: int = 5                    # ticks a post stays in candidate inboxes
    rho_s: float = 0.9                        # discourse attention decay
    rho_sigma: float = 0.9                    # dominant stance decay
    cascade_depth_decay: float = 0.7          # rho^depth visibility
    max_cascade_depth: int = 4
    max_cascade_size: int = 1000              # warning threshold, per tick

    drift: str = "full"                       # none | social | full
    drift_lr: float = 0.02                    # reinforcement channel
    drift_lr_social: float = 0.01             # social influence channel
    ou_k: tuple[tuple[str, float], ...] = ()  # (block, rate) overrides
    noise_sigma: float = 0.002
    llm_adjudication: bool = False            # queued only; offline pass in v1

    snapshot_every: int = 1
    exposure_sample_rate: float = 0.01


@dataclass(frozen=True)
class ScenarioConfig(Hashable):
    """Scenario layer, compatible with the stance editor's emitted JSON.

    Each axis: {name, pole_neg, pole_pos, marginal: {kind: empirical, bins,
    support, density}, expression_cost: {neg, pos}}.
    """

    name: str = "default"
    stance_axes: tuple[dict, ...] = ()
    topic_names: tuple[str, ...] = ()

    def __post_init__(self):
        for ax in self.stance_axes:
            d = ax.get("marginal", {}).get("density")
            if d is None:
                raise ValueError(f"axis {ax.get('name')!r} lacks an empirical marginal")
        if self.topic_names and len(self.topic_names) != len(set(self.topic_names)):
            raise ValueError("topic names must be unique")

    @classmethod
    def from_editor_json(cls, data: dict, name: str = "scenario") -> "ScenarioConfig":
        axes = tuple(data["scenario"]["stance_axes"])
        return cls(name=name, stance_axes=axes)

    def axis_count(self) -> int:
        return len(self.stance_axes)


@dataclass(frozen=True)
class Config(Hashable):
    population: PopulationConfig = field(default_factory=PopulationConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    label: str = "default"

    def stance_dims(self) -> int:
        n = self.scenario.axis_count()
        if n > 0:
            return n
        return max(1, self.population.stance_dims)

    def sub_hashes(self) -> dict[str, str]:
        return {
            "population": self.population.hash(),
            "graph": self.graph.hash(),
            "dynamics": self.dynamics.hash(),
            "scenario": self.scenario.hash(),
        }
