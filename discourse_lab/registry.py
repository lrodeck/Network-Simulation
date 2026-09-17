"""Name-keyed component registry.

Registration decorators run at module load (dev notes §3.1). Iteration order is
always sorted so that registry order can never leak into results.

Change spec V8.3 (the linked-metric registry): two instances of a reported
metric being a reparameterisation of another metric it was then compared
against as if independent — `recovery_fraction`/`split_ratio` (see
`experiments/experiment03_bubble_intervention.py`) and, less cleanly, the
`toward_mean`/`toward_own_pole`, `algorithmic_share`/`rank_penalty` pairs
(`outcomes.py`) and `ingroup`/`outgroup` (`dynamics/drift.py::affect_delta`,
`mode="distance"`) — were each written up as findings before being caught.
`register(..., derives_from=...)` records a clean parent/child relation (a
metric that is an algebraic function of another registered metric or of a
config constant); `ALGEBRAIC_LINKS` records the messier cross-metric
identities that are not a clean parent/child relation. `check_comparison`
raises on either, so a table-building helper cannot silently present a
linked pair as two independent readings again.
"""

from __future__ import annotations

from typing import Any, Callable

_TABLES: dict[str, dict[str, Any]] = {}
_DERIVES_FROM: dict[tuple[str, str], tuple[str, ...]] = {}

# Cross-metric identities that are NOT a clean parent/child relation (see
# module docstring). Keyed by an unordered pair so `check_comparison` does
# not care which order the caller passes them in. Seeded with the three
# known links of this shape; `recovery_fraction`/`split_ratio` is the fourth
# known link but IS a clean parent/child and is registered via
# `derives_from` instead (see `experiments/experiment03_bubble_intervention.py`).
ALGEBRAIC_LINKS: dict[frozenset, str] = {
    frozenset({"ingroup", "outgroup"}):
        "ingroup == 1 - outgroup (dynamics/drift.py::affect_delta, "
        "mode='distance') -- any animus/identification coupling metric "
        "built from both is partly structural, not two independent reads",
    frozenset({"toward_mean", "toward_own_pole"}):
        "both are movement along the SAME fixed pre-period dominant axis "
        "(experiments/experiment03_bubble_intervention.py::_ideo_decomposition) "
        "-- toward_mean as Euclidean distance to the pre-period mean, "
        "toward_own_pole as signed projection onto that axis -- not an exact "
        "function of one another, but a mirror pair, not independent evidence",
    frozenset({"algorithmic_share", "rank_penalty"}):
        "both decompose the SAME ranked candidate pool "
        "(outcomes.py::cross_cutting_exposure); rank_penalty is the measure "
        "that survives where algorithmic_share degenerates to all-NaN at "
        "inject_k == 0, not an independently informative second quantity",
}


class LinkedMetricError(ValueError):
    """Raised by `check_comparison` when two metrics are known to be linked."""


def register(kind: str, name: str, derives_from: tuple[str, ...] = ()) -> Callable:
    def deco(obj: Any) -> Any:
        table = _TABLES.setdefault(kind, {})
        if name in table:
            raise ValueError(f"duplicate registration: {kind}/{name}")
        table[name] = obj
        if derives_from:
            _DERIVES_FROM[(kind, name)] = tuple(derives_from)
        return obj

    return deco


def get(kind: str, name: str) -> Any:
    try:
        return _TABLES[kind][name]
    except KeyError:
        available = names(kind)
        raise KeyError(f"unknown {kind} {name!r}; available: {available}") from None


def names(kind: str) -> list[str]:
    return sorted(_TABLES.get(kind, {}))


def derives_from(kind: str, name: str) -> tuple[str, ...]:
    """Other registered outcomes or config constants `name` is an algebraic
    function of, as named in its `register(..., derives_from=...)` call."""
    return _DERIVES_FROM.get((kind, name), ())


def check_comparison(metric_a: str, metric_b: str) -> None:
    """Raise if `metric_a`/`metric_b` are known to be algebraically linked —
    directly (`ALGEBRAIC_LINKS`) or via a registered parent/child
    (`derives_from`, in any kind) — so a helper that builds an experiment
    comparison table cannot silently compare a reparameterisation against
    the metric it was reparameterised from.

    Raising, not warning: a warning here would be ignored on the evidence
    of every other warning in this codebase, and a notebook cell comparing
    a linked pair is a bug, not a style choice to flag and move past.
    """
    if metric_a == metric_b:
        return
    pair = frozenset({metric_a, metric_b})
    if pair in ALGEBRAIC_LINKS:
        raise LinkedMetricError(
            f"{metric_a!r} and {metric_b!r} are algebraically linked "
            f"({ALGEBRAIC_LINKS[pair]}) -- comparing them as independent "
            "evidence is circular"
        )
    for (kind, child), parents in _DERIVES_FROM.items():
        if child == metric_a and metric_b in parents:
            raise LinkedMetricError(
                f"{metric_a!r} derives from {metric_b!r} ({kind}/{child} is "
                f"registered with derives_from={parents}) -- comparing them "
                "as independent evidence is circular"
            )
        if child == metric_b and metric_a in parents:
            raise LinkedMetricError(
                f"{metric_b!r} derives from {metric_a!r} ({kind}/{child} is "
                f"registered with derives_from={parents}) -- comparing them "
                "as independent evidence is circular"
            )
