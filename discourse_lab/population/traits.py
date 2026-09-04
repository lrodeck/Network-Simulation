"""Trait table (spec §1.1): the column layout of `X`, block by block, with
each column's target marginal and stored-space link (spec §1.1, §2.1).

Traits without an explicit distribution in spec §2.1 (most of expression and
behavior) default to a symmetric Beta(2,2) if bounded in [0,1], or N(0,1) if
unbounded — the spec only pins down the traits that drive heavy-tailed or
skewed dynamics (activity, prominence, contrarianism, plasticity, conviction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discourse_lab.config import Config
from discourse_lab.population.marginals import Marginal, build_marginal, empirical_from_editor

PERSONALITY = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
EXPRESSION = ("verbosity", "formality", "irony", "humor", "profanity", "emoji")
BEHAVIOR = ("activity", "reply_prop", "repost_prop", "contrarianism", "credulity", "prominence")
META = ("plasticity", "conviction", "circadian_phase")


@dataclass(frozen=True)
class TraitSpec:
    name: str
    block: str
    marginal: Marginal
    link: str  # "identity" | "logit" | "log"


def _behavior_marginal(name: str, cfg: Config) -> tuple[Marginal, str]:
    pop = cfg.population
    if name == "activity":
        return build_marginal("lognormal", sigma=pop.activity_sigma), "log"
    if name == "prominence":
        return build_marginal("pareto", alpha=pop.pareto_alpha), "log"
    if name == "contrarianism":
        return build_marginal("beta", a=2.0, b=5.0), "logit"
    if name in ("reply_prop", "repost_prop", "credulity"):
        return build_marginal("beta", a=2.0, b=2.0), "logit"
    raise KeyError(name)


def _meta_marginal(name: str) -> tuple[Marginal, str]:
    if name == "plasticity":
        return build_marginal("beta", a=2.0, b=8.0), "logit"
    if name == "conviction":
        return build_marginal("beta", a=5.0, b=2.0), "logit"
    if name == "circadian_phase":
        return build_marginal("vonmises", mu=0.0, kappa=2.0), "identity"
    raise KeyError(name)


def stance_specs(cfg: Config) -> list[TraitSpec]:
    axes = cfg.scenario.stance_axes
    d = cfg.stance_dims()
    if axes:
        specs = []
        for i, ax in enumerate(axes):
            m = ax["marginal"]
            marginal = empirical_from_editor(bins=m["bins"], support=tuple(m["support"]), density=m["density"])
            specs.append(TraitSpec(f"stance_{ax.get('name', i)}", "stance", marginal, "identity"))
        return specs
    return [
        TraitSpec(f"stance_{i}", "stance", build_marginal("normal"), "identity") for i in range(d)
    ]


def trait_table(cfg: Config) -> list[TraitSpec]:
    specs: list[TraitSpec] = []

    for name in PERSONALITY:
        specs.append(TraitSpec(name, "personality", build_marginal("normal"), "identity"))

    for name in EXPRESSION:
        specs.append(TraitSpec(name, "expression", build_marginal("beta", a=2.0, b=2.0), "logit"))

    for k in range(cfg.population.n_topics):
        marginal = build_marginal("normal", sigma=cfg.population.topic_logit_sigma)
        specs.append(TraitSpec(f"topic_affinity_{k}", "topic_affinity", marginal, "identity"))

    specs.extend(stance_specs(cfg))

    for name in BEHAVIOR:
        marginal, link = _behavior_marginal(name, cfg)
        specs.append(TraitSpec(name, "behavior", marginal, link))

    for name in META:
        marginal, link = _meta_marginal(name)
        specs.append(TraitSpec(name, "meta", marginal, link))

    return specs


def trait_names(cfg: Config) -> list[str]:
    return [s.name for s in trait_table(cfg)]
