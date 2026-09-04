"""Name-keyed component registry.

Registration decorators run at module load (dev notes §3.1). Iteration order is
always sorted so that registry order can never leak into results.
"""

from __future__ import annotations

from typing import Any, Callable

_TABLES: dict[str, dict[str, Any]] = {}


def register(kind: str, name: str) -> Callable:
    def deco(obj: Any) -> Any:
        table = _TABLES.setdefault(kind, {})
        if name in table:
            raise ValueError(f"duplicate registration: {kind}/{name}")
        table[name] = obj
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
