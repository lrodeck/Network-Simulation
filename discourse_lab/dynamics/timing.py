"""Activity and timing (spec §2.3): a two-peak circadian shape, Poisson
posting with fatigue.

    lambda_u(t) = activity_u * circ(t - phi_u) * fatigue_u(t)
    n_posts_u(t) ~ Poisson(lambda_u(t) * dt)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def circadian_shape(ticks_per_day: int, morning: float = 0.38, evening: float = 0.79, kappa: float = 6.0) -> np.ndarray:
    """A fixed two-peak diurnal shape over one day, mean-normalised to 1 so it
    multiplies the base activity rate without shifting its long-run mean.

    `morning`/`evening` are peak times as a fraction of the day (~9am, ~7pm);
    `kappa` sets how sharp each peak is (von Mises concentration).
    """
    phase = 2 * np.pi * np.arange(ticks_per_day) / ticks_per_day
    peak_a = 2 * np.pi * morning
    peak_b = 2 * np.pi * evening
    shape = np.exp(kappa * np.cos(phase - peak_a)) + np.exp(kappa * np.cos(phase - peak_b))
    shape = shape / shape.mean()
    return shape


def circadian_factor(t: int, ticks_per_day: int, phase_u: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """circ(t - phi_u) for every user, `phase_u` in ticks (their personal
    circadian offset, spec §1.1 meta trait `circadian_phase` converted to a
    tick offset by the caller).
    """
    idx = (t - phase_u).astype(int) % ticks_per_day
    return shape[idx]


@dataclass
class FatigueState:
    """Multiplicative suppression after a posting burst: fatigue decays back
    toward 1 each tick and drops when a user posts.
    """

    level: np.ndarray  # 1.0 = fully rested, -> 0 as recent posting accumulates

    @classmethod
    def initial(cls, n_users: int) -> "FatigueState":
        return cls(level=np.ones(n_users))

    def factor(self) -> np.ndarray:
        return self.level

    def step(self, n_posts: np.ndarray, decay: float) -> None:
        self.level *= decay
        self.level += (1 - decay) * 1.0  # relax back toward 1
        self.level /= 1.0 + n_posts  # burst suppression, felt next tick


def sample_post_counts(
    rng: np.random.Generator,
    activity: np.ndarray,
    circ: np.ndarray,
    fatigue: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    rate = np.clip(activity * circ * fatigue * dt, 0, None)
    return rng.poisson(rate)
