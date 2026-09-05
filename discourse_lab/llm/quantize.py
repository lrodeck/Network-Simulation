"""Traits -> prompt (spec §2.10): never put raw floats in a prompt. Floats
produce a model that ignores them; verbal labels produce one that acts on
them. Quantize each value to an `n_bands`-point band using the population's
own quantiles as the reference distribution, so "high" means "high relative
to this population," not an arbitrary fixed cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_KNOWN_LABELS: dict[int, tuple[str, ...]] = {
    3: ("low", "medium", "high"),
    5: ("very low", "low", "medium", "high", "very high"),
    7: ("very low", "low", "somewhat low", "medium", "somewhat high", "high", "very high"),
}


def band_labels(n_bands: int) -> tuple[str, ...]:
    return _KNOWN_LABELS.get(n_bands, tuple(f"band_{i}" for i in range(n_bands)))


@dataclass
class Bands:
    """Fitted quantile edges for a set of columns, ready to label any value."""

    labels: tuple[str, ...]
    edges: dict[str, np.ndarray]  # name -> (n_bands - 1,) quantile edges

    @classmethod
    def fit(cls, values_by_name: dict[str, np.ndarray], n_bands: int = 5) -> "Bands":
        quantiles = np.linspace(0, 1, n_bands + 1)[1:-1]
        edges = {name: np.quantile(v, quantiles) for name, v in values_by_name.items()}
        return cls(labels=band_labels(n_bands), edges=edges)

    def label(self, name: str, value: float) -> str:
        idx = int(np.searchsorted(self.edges[name], value))
        return self.labels[idx]

    def label_row(self, values: dict[str, float]) -> dict[str, str]:
        return {name: self.label(name, v) for name, v in values.items()}
