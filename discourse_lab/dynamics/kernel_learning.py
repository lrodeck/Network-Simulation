"""Learnable kernels (change spec C3b): three tiers, three named rules.

A toolbox should let you swap the *learning rule* the way it lets you swap
the kernel. But "learnable theta" is not one theory, and full per-user theta
has three real costs (an (N, |actions|, |features|) tensor, a gather over
every exposure, and a jump from ~17 free coefficients to N x 17 that makes
any macro pattern reachable and guts separability analysis). So:

    none        theta is a table you can read — every pre-C3 result
    group_gain  one multiplicative gain per user per named coefficient group
                (theta_u = theta_base * g_u) — the learnable default
    full        coefficient-granular gains, unrestricted theta_u — shipped
                for honesty about the toolbox claim, expected to be rarely
                used

Two constraints from the change spec, both load-bearing here:

1. Learning never replaces a named kernel — it modulates one. A run must
   always be describable as "outrage, plus this much learned deviation",
   which is why the state starts at g = 1 (exactly the fixed-theta kernel)
   and reports drift away from it.
2. The learning rule is named in the config, never implied, because the
   reward signal is genuinely underdetermined:

       conformity   gains drift toward the revealed engagement behaviour of
                    in-neighbours — Brady et al. (2021)'s norm convergence;
                    the only rule that makes the network matter
       bandit       gains drift toward what preceded above-baseline
                    engagement on the user's own posts — reinforcement in
                    the broad sense
       habituation  features the user engaged with amplify by use, others
                    decay — mere exposure, no social channel. THE CONTROL:
                    use-driven amplification alone produces some apparent
                    norm convergence, and the social claim is the difference
                    between it and conformity.

The state is kernel-owned, NOT a trait-matrix block: the population artifact
is cached on a structural sub-hash independently of the kernel, and coupling
them would invalidate the population cache on every kernel swap. That
duplicates a little drift scaffolding; the alternative duplicates the
population cache across every kernel in a sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from discourse_lab.exposure.kernel import LEARNED_GROUPS
from discourse_lab.registry import register


@dataclass
class KernelLearningState:
    """Per-user gains over coefficient groups, plus the author-side running
    engagement baseline the bandit rule needs for "above baseline", and the
    per-user EMA of *revealed engagement behaviour* (`b`) the conformity
    rule converges toward.

    Invariant: `g == 1` reproduces the named fixed-theta kernel exactly, so
    every learned run is anchored to a readable theory.
    """

    n_users: int
    groups: tuple[str, ...] = LEARNED_GROUPS
    g: np.ndarray | None = field(default=None)
    engagement_baseline: np.ndarray | None = field(default=None)
    baseline_ema: float = 0.2
    behaviour_ema: float = 0.2
    b: np.ndarray | None = field(default=None)

    def ensure_initialized(self) -> None:
        if self.g is None:
            self.g = np.ones((self.n_users, len(self.groups)))
        if self.engagement_baseline is None:
            self.engagement_baseline = np.zeros(self.n_users)
        if self.b is None:
            self.b = np.zeros((self.n_users, len(self.groups)))

    def group_index(self, group: str) -> int:
        return self.groups.index(group)

    def observe_behaviour(self, users: np.ndarray, profile: np.ndarray) -> None:
        """Roll this tick's engaged exposures into the per-user behaviour EMA.

        `b` is what the conformity rule converges toward — a user's revealed
        engagement profile, NOT their gains. Gains converging toward gains
        has no source term: everyone starts at exactly 1 and the rule can
        never move anything. Behaviour is the signal; gains follow it.
        """
        self.ensure_initialized()
        counts = np.zeros(self.n_users)
        acc = np.zeros_like(self.b)
        np.add.at(counts, users, 1.0)
        np.add.at(acc, users, profile)
        observed = counts > 0
        mean_now = acc / np.maximum(counts, 1)[:, None]
        self.b[observed] = (
            (1 - self.behaviour_ema) * self.b[observed]
            + self.behaviour_ema * mean_now[observed]
        )

    def author_surprise(self, author: np.ndarray, engagement_delta: np.ndarray) -> np.ndarray:
        """Per-post surprise against the author's own running baseline — the
        same r_p reading as drift channel 1, kept separate so the kernel
        rules' reward signal does not perturb the trait channels' baseline.
        """
        self.ensure_initialized()
        surprise = engagement_delta - self.engagement_baseline[author]
        n = len(self.engagement_baseline)
        per_author_sum = np.zeros(n)
        counts = np.zeros(n)
        np.add.at(per_author_sum, author, engagement_delta)
        np.add.at(counts, author, 1.0)
        posted = counts > 0
        per_author_mean = np.divide(per_author_sum, counts, out=np.zeros(n), where=posted)
        posted_idx = np.flatnonzero(posted)
        self.engagement_baseline[posted_idx] = (
            (1 - self.baseline_ema) * self.engagement_baseline[posted_idx]
            + self.baseline_ema * per_author_mean[posted_idx]
        )
        return surprise


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardize a feature column so group profiles are comparable across
    features with wildly different scales (log1p(social_proof) vs agreement)."""
    sd = x.std()
    if sd < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def _group_profile(features: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    """(m, G) standardized group-profile matrix over exposure rows."""
    from discourse_lab.exposure.kernel import FEATURE_GROUPS

    out = np.zeros((len(rows), len(LEARNED_GROUPS)))
    for gi, group in enumerate(LEARNED_GROUPS):
        members = FEATURE_GROUPS[group]
        cols = [_zscore(features[f]) for f in members if f in features]
        if cols:
            out[:, gi] = np.mean(cols, axis=0)[rows]
    return out


@register("kernel_learning_rule", "conformity")
def conformity_rule(state: KernelLearningState, ctx: dict) -> None:
    """Gains drift toward the *revealed engagement behaviour* of followees —
    the EMA'd group profile of what each followee actually engaged with
    (`state.b`). Users whose followees have engaged nothing this window keep
    their gains: absence of evidence reads as reversion toward the anchored
    kernel, not as evidence for the population mean."""
    graph = ctx["graph"]
    lr = ctx["lr_kernel"]
    state.ensure_initialized()
    csr = graph.csr
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    if not (row_sums > 0).any():
        return
    # row-mean of followee behaviour: (csr @ b) / outdegree
    target = np.asarray(csr @ state.b) / np.maximum(row_sums, 1)[:, None]
    mask = (row_sums > 0)[:, None]
    state.g += ctx["ramp"] * lr * mask * (target - state.g)


@register("kernel_learning_rule", "bandit")
def bandit_rule(state: KernelLearningState, ctx: dict) -> None:
    """For each of a user's own posts that over-performed its baseline, that
    author's gains drift toward the group profile of the exposure pairs that
    actually engaged with it — a longer, weaker causal path than
    conformity: the reward arrives through the audience, not the norm."""
    exposures = ctx["exposures"]
    posts = ctx["posts"]
    engagement_delta = ctx["engagement_delta"]
    lr = ctx["lr_kernel"]
    state.ensure_initialized()

    engaged = ctx["engaged"]
    if not engaged.any() or engagement_delta is None or len(posts) == 0:
        return

    post_idx = exposures.post_idx[engaged]
    author = posts.author[post_idx]
    surprise_post = state.author_surprise(np.arange(len(posts)), engagement_delta)
    # tanh keeps one runaway post from dominating the gain update
    weight = np.tanh(surprise_post[post_idx])[:, None]

    profile = ctx["profile"][engaged]

    counts = np.zeros(state.n_users)
    acc = np.zeros((state.n_users, len(state.groups)))
    np.add.at(counts, author, 1.0)
    np.add.at(acc, author, weight * profile)
    hit = counts > 0
    profile_mean = acc / np.maximum(counts, 1)[:, None]
    state.g[hit] += ctx["ramp"] * lr * (profile_mean[hit] - state.g[hit])


@register("kernel_learning_rule", "habituation")
def habituation_rule(state: KernelLearningState, ctx: dict) -> None:
    """THE CONTROL. Features a user engaged with amplify by use; features
    they skipped decay. No social term anywhere — standing to `conformity`
    as `null` stands to the kernels. If conformity's variance collapse also
    appears here, the social claim is unsupported."""
    exposures = ctx["exposures"]
    lr = ctx["lr_kernel"]
    state.ensure_initialized()

    engaged = ctx["engaged"]
    if len(exposures) == 0:
        return

    u = exposures.user_id
    profile = ctx["profile"]

    counts = np.zeros(state.n_users)
    acc = np.zeros((state.n_users, len(state.groups)))
    np.add.at(counts, u[engaged], 1.0)
    np.add.at(acc, u[engaged], profile[engaged])
    # skipped exposures decay the groups they were served in
    skipped = ~engaged
    skip_counts = np.zeros(state.n_users)
    skip_acc = np.zeros((state.n_users, len(state.groups)))
    np.add.at(skip_counts, u[skipped], 1.0)
    np.add.at(skip_acc, u[skipped], profile[skipped])

    engaged_mean = np.divide(acc, np.maximum(counts, 1)[:, None])
    skipped_mean = np.divide(skip_acc, np.maximum(skip_counts, 1)[:, None])
    seen = (counts + skip_counts) > 0
    state.g[seen] += ctx["ramp"] * lr * (engaged_mean[seen] - skipped_mean[seen])


def apply_learning(
    state: KernelLearningState,
    cfg,
    graph,
    exposures,
    actions,
    posts,
    features: dict[str, np.ndarray],
    engagement_delta: np.ndarray | None,
    t: int,
    rng: np.random.Generator,
) -> None:
    """One tick of kernel learning: the named rule, then OU reversion toward
    g = 1. Reversion is what keeps a learnable kernel the theory it claims
    to be (change spec C3b: "a learnable kernel that wanders arbitrarily far
    from its named base is no longer the theory it claims to be")."""
    from discourse_lab.dynamics.drift import ramp_factor
    from discourse_lab.registry import get

    state.ensure_initialized()
    engaged = actions != "skip"
    rows = np.arange(len(exposures.user_id))
    # every rule reads the same standardized group profile; the behaviour EMA
    # is updated here so conformity's signal is fresh regardless of rule
    profile = _group_profile(features, rows)
    if engaged.any():
        state.observe_behaviour(exposures.user_id[engaged], profile[engaged])

    ctx = {
        "graph": graph,
        "exposures": exposures,
        "actions": actions,
        "posts": posts,
        "features": features,
        "profile": profile,
        "rows": rows,
        "engaged": engaged,
        "engagement_delta": engagement_delta,
        "lr_kernel": cfg.dynamics.lr_kernel,
        "ramp": ramp_factor(t, cfg.dynamics.drift_ramp_ticks),
        "rng": rng,
    }
    get("kernel_learning_rule", cfg.dynamics.kernel_learning_rule)(state, ctx)
    # OU reversion toward the anchored kernel
    state.g += cfg.dynamics.kernel_gain_ou_k * (1.0 - state.g)
