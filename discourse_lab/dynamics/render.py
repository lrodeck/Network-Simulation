"""Stub renderer (dev §6 step 5): a placeholder text form for a post before
the real LLM realization pass (step 12) exists, so downstream code has
something to print/log against.
"""

from __future__ import annotations

import numpy as np

from discourse_lab.dynamics.posts import PostBatch

_BANDS = ("mild", "measured", "engaged", "strong", "intense")


def _band(intensity: np.ndarray) -> np.ndarray:
    edges = np.linspace(0, 1, len(_BANDS) + 1)[1:-1]
    idx = np.digitize(intensity, edges)
    return np.array(_BANDS)[idx]


def stub_render(posts: PostBatch) -> list[str]:
    intensity = (posts.arousal + posts.provocativeness) / 2
    labels = _band(intensity)
    return [
        f"[u{a} - topic{t} - {label}]"
        for a, t, label in zip(posts.author, posts.topic, labels)
    ]
