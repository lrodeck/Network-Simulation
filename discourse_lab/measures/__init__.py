"""Measure registry (dev §6 step 7, §7.1): tick-level diagnostics the run
monitor plots live and post-run analysis reads back. Each measure is
registered by name so `metrics.parquet` columns and the monitor's plots stay
in sync with whatever is actually computed.
"""

from __future__ import annotations

import numpy as np

from discourse_lab.dynamics.perception import PerceivedState
from discourse_lab.registry import get, names, register


@register("measure", "salience_stance_agreement")
def salience_stance_agreement(perceived: PerceivedState, stance_u: np.ndarray) -> tuple[float, float]:
    """The salience/stance agreement pair (dev §7.1): for each user's own
    most-salient perceived topic, how dominant it feels (`salience`) and how
    much their own stance agrees with the perceived dominant stance there
    (`agreement`, higher = more agreement). Returned as tick-level means.
    """
    n = perceived.s_perceived.shape[0]
    k_star = perceived.s_perceived.argmax(axis=1)
    salience = perceived.s_perceived[np.arange(n), k_star]

    dominant_stance = perceived.sigma_perceived[np.arange(n), k_star]
    agreement = -np.linalg.norm(stance_u - dominant_stance, axis=1)

    return float(salience.mean()), float(agreement.mean())


@register("measure", "bubble_index")
def bubble_index(perceived: PerceivedState, global_stance_var: float) -> float:
    """How homogeneous each user's perceived environment is relative to the
    population as a whole: `1 - local_variance / global_variance`, where
    local variance is the variance of stance *among the posts a user
    actually saw* (`sigma_var_local`), averaged over topics they had at least
    two exposures on (variance needs >=2 points; users with none contribute
    nothing). Clipped to [0, 1]; 1 means a user's feed shows no stance
    diversity at all where the population has plenty (an echo chamber), 0
    means their feed is exactly as diverse as the population.
    """
    seen_topics = perceived.s_local > 0
    if not seen_topics.any():
        return float("nan")

    local_var = perceived.sigma_var_local[seen_topics]
    ratio = local_var / max(global_stance_var, 1e-9)
    per_user_topic = np.clip(1 - ratio, 0, 1)
    return float(per_user_topic.mean())


def _gini(x: np.ndarray) -> float:
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    sorted_x = np.sort(x.astype(float))
    n = len(sorted_x)
    cum = np.cumsum(sorted_x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


@register("measure", "attention_gini")
def attention_gini(engagement_count: np.ndarray) -> float:
    """Gini coefficient of engagement counts across posts this tick: 0 =
    perfectly even attention, -> 1 as it concentrates on a handful of posts.
    """
    return _gini(engagement_count)


def measure_names() -> list[str]:
    return names("measure")


def compute_measure(name: str, *args, **kwargs):
    return get("measure", name)(*args, **kwargs)
