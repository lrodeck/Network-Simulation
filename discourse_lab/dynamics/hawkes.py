"""Hawkes replies (spec §2.3, §3): reply intensity to a post is self-exciting,
not Poisson, which is what produces realistic thread burstiness — a post
gets its comments in a clump, not spread uniformly.

    lambda_p(t) = mu_p + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

`alpha/beta` (`hawkes_ratio`) must stay < 1 for stability; as it approaches 1
you get pile-on dynamics. Implemented with the standard exponential-kernel
recursion so per-tick cost is O(active threads), not O(events).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HawkesThreads:
    """One entry per open thread (root post). `excitation` is the decayed sum
    of past events (the recursive Hawkes state); `age` is ticks since the
    root post, used to close threads past `max_thread_age`.
    """

    post_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    mu: np.ndarray = field(default_factory=lambda: np.empty(0))
    excitation: np.ndarray = field(default_factory=lambda: np.empty(0))
    age: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def open_thread(self, post_id: int, mu: float) -> None:
        self.post_id = np.append(self.post_id, post_id)
        self.mu = np.append(self.mu, mu)
        self.excitation = np.append(self.excitation, 0.0)
        self.age = np.append(self.age, 0)

    def step(self, rng: np.random.Generator, alpha: float, beta: float, max_age: int, dt: float = 1.0) -> dict[int, int]:
        """Advance one tick: decay, draw replies, excite, age out. Returns
        {post_id: n_replies} for threads that got at least one reply.
        """
        if len(self.post_id) == 0:
            return {}

        self.excitation *= np.exp(-beta * dt)
        intensity = self.mu + alpha * self.excitation
        n_replies = rng.poisson(np.clip(intensity * dt, 0, None))
        self.excitation += n_replies.astype(float)
        self.age += 1

        keep = self.age < max_age
        result = {int(pid): int(n) for pid, n in zip(self.post_id, n_replies) if n > 0}

        self.post_id, self.mu = self.post_id[keep], self.mu[keep]
        self.excitation, self.age = self.excitation[keep], self.age[keep]
        return result

    def __len__(self) -> int:
        return len(self.post_id)
