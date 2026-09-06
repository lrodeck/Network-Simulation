"""Names for the things the model computes over.

Two different stance column namings exist and the difference is a live trap:

    pop.trait_names     stance_{axis_name}   e.g. "stance_institutional trust"
    posts.parquet       stance_{d}           e.g. "stance_2"

`io/store.py` flattens stance positionally because `pa.array` rejects a 2-D
array, while `population/traits.py:stance_specs` names trait columns after the
scenario's axes — and the packaged default has an axis called "institutional
trust", *with a space*. So `name.startswith("stance_")` is ambiguous across the
two surfaces, and anything that assumes `stance_0` breaks the moment a scenario
is attached. The Lexicon owns both mappings so no caller has to know.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Lexicon:
    axis_names: tuple[str, ...]
    axis_poles: tuple[tuple[str, str], ...]   # (negative, positive) per axis
    topic_names: tuple[str, ...]
    scenario_name: str = "unnamed"

    # -- naming surfaces ----------------------------------------------------

    def trait_column(self, d: int) -> str:
        """Column name in `pop.trait_names` / the traits table."""
        return f"stance_{self.axis_names[d]}"

    def post_column(self, d: int) -> str:
        """Column name in `posts.parquet` — always positional."""
        return f"stance_{d}"

    def axis_label(self, d: int) -> str:
        neg, pos = self.axis_poles[d]
        return f"{self.axis_names[d]} ({neg} → {pos})"

    def pole_label(self, d: int, sign: float) -> str:
        """Which pole a signed stance value leans toward."""
        neg, pos = self.axis_poles[d]
        return pos if sign >= 0 else neg

    def topic_label(self, k: int) -> str:
        if k < len(self.topic_names):
            return self.topic_names[k]
        return f"topic {k}"

    def stance_columns(self, names: Sequence[str]) -> list[int]:
        """Indices of the stance columns in `names`, in axis order.

        Exact-name resolution against `trait_column`, falling back to the
        positional form, so it works on either surface without the caller
        knowing which one it holds.
        """
        lookup = {name: i for i, name in enumerate(names)}
        out = []
        for d in range(len(self.axis_names)):
            for candidate in (self.trait_column(d), self.post_column(d)):
                if candidate in lookup:
                    out.append(lookup[candidate])
                    break
            else:
                raise KeyError(
                    f"no column for stance axis {d} ({self.axis_names[d]!r}); "
                    f"looked for {self.trait_column(d)!r} and {self.post_column(d)!r}"
                )
        return out

    @property
    def n_axes(self) -> int:
        return len(self.axis_names)

    # -- construction -------------------------------------------------------

    @classmethod
    def generic(cls, d: int, k: int) -> "Lexicon":
        """Positional fallback when no scenario is attached: `stance_0`,
        `topic 3`. Everything still works, it just reads worse."""
        return cls(
            axis_names=tuple(str(i) for i in range(d)),
            axis_poles=tuple(("-", "+") for _ in range(d)),
            topic_names=(),
        )

    @classmethod
    def from_config(cls, cfg) -> "Lexicon":
        scenario = cfg.scenario
        if scenario.axis_count() == 0:
            return cls.generic(cfg.stance_dims(), cfg.population.n_topics)
        return cls(
            axis_names=scenario.axis_names(),
            axis_poles=scenario.poles(),
            topic_names=tuple(scenario.topic_names),
            scenario_name=scenario.name,
        )

    @classmethod
    def from_handle(cls, handle) -> "Lexicon":
        """Rebuild from a persisted run — `meta.json` carries the whole config,
        so post-run labelling needs no extra file."""
        import json

        from discourse_lab.config import Config

        raw = handle.meta.get("config_json")
        if raw:
            return cls.from_config(Config.from_json(raw) if hasattr(Config, "from_json")
                                   else _config_from_dict(json.loads(raw)))
        return cls.generic(1, 0)


def _config_from_dict(data: dict):
    """Minimal shim: only the scenario and topic count are needed for naming,
    so this avoids depending on a full Config deserialiser existing."""
    from discourse_lab.config import Config, PopulationConfig, ScenarioConfig

    scenario = data.get("scenario") or {}
    population = data.get("population") or {}
    return Config(
        population=PopulationConfig(
            n_topics=int(population.get("n_topics", 8)),
            stance_dims=int(population.get("stance_dims", 3)),
        ),
        scenario=ScenarioConfig(
            name=str(scenario.get("name", "unnamed")),
            stance_axes=tuple(scenario.get("stance_axes", ())),
            topic_names=tuple(scenario.get("topic_names", ())),
        ),
    )


_CACHE: dict[str, Lexicon] = {}


def lexicon_for(cfg) -> Lexicon:
    """Memoised on `cfg.hash()`.

    Not `functools.lru_cache(cfg)`: `Config` is a frozen dataclass but its
    scenario holds plain dicts, so it is not hashable and the decorator raises
    at the first call. The hash string is the key.
    """
    key = cfg.hash()
    hit = _CACHE.get(key)
    if hit is None:
        hit = Lexicon.from_config(cfg)
        _CACHE[key] = hit
    return hit
