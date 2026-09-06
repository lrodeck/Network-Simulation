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


@dataclass
class HawkesThreads:
    """Self-exciting reply intensity per post (spec §2.4).

    A post that has just been replied to is more likely to be replied to
    again — this is what makes discourse threads *deep* rather than merely
    wide. Reposts fan a root out to new audiences and stop; reply chains are
    where depth comes from, and they are self-exciting: an argument in
    progress attracts more argument.

        lambda_p(t) = mu0 + sum_{replies i to p, t_i < t} alpha * exp(-beta * (t - t_i))

    stored as the excess over `mu0` and advanced recursively, so the cost per
    tick is O(open threads) rather than O(all replies ever):

        excess(t) = excess(t-1) * exp(-beta * dt) + alpha * (new replies)

    `alpha = ratio * beta` keeps the branching ratio `alpha / beta = ratio`,
    which must stay below 1 for the process to be subcritical — above it,
    reply chains never terminate.

    Without this, a reply lands wherever the follower graph happens to
    re-expose a post, which is depth-agnostic: measured thread depth sat at
    ~1.2-1.4 against spec §5.1's 1.5-3 no matter how the kernel intercepts
    were set, and raising `post_lifetime` from 5 to 25 ticks moved it by
    0.03. Time was never the constraint; targeting was.
    """

    excess: dict[int, float]      # post id -> intensity above mu0
    opened_t: dict[int, int]      # post id -> tick the thread opened

    @classmethod
    def empty(cls) -> "HawkesThreads":
        return cls(excess={}, opened_t={})

    def step(self, t: int, beta: float, max_thread_age: int) -> None:
        """Decay every open thread one tick and close the stale ones."""
        decay = float(np.exp(-beta))
        for post_id in list(self.excess):
            if t - self.opened_t[post_id] > max_thread_age:
                del self.excess[post_id]
                del self.opened_t[post_id]
                continue
            self.excess[post_id] *= decay

    def intensity(self, post_ids: np.ndarray, mu0: float) -> np.ndarray:
        """Current reply intensity for each post, `mu0` for unopened ones."""
        excess = self.excess
        return mu0 + np.fromiter(
            (excess.get(int(p), 0.0) for p in post_ids), dtype=np.float64, count=len(post_ids)
        )

    def record(
        self,
        parent_ids: np.ndarray,
        child_ids: np.ndarray,
        t: int,
        ratio: float,
        beta: float,
    ) -> None:
        """Register this tick's replies: excite each parent, and open each new
        reply post as a thread of its own.

        The child inherits a `ratio`-scaled share of its parent's excitement
        rather than starting cold — a reply inside a heated exchange is itself
        a likely target. That inheritance is what carries activity *down* a
        chain and so produces depth; scaling by `ratio < 1` is what stops it
        running away.
        """
        alpha = ratio * beta
        for parent in parent_ids:
            parent = int(parent)
            self.excess[parent] = self.excess.get(parent, 0.0) + alpha
            self.opened_t.setdefault(parent, t)

        for parent, child in zip(parent_ids, child_ids):
            child = int(child)
            self.excess[child] = ratio * self.excess.get(int(parent), 0.0)
            self.opened_t[child] = t
