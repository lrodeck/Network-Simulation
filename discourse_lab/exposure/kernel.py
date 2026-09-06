"""The engagement kernel (spec §2.6): a multinomial logit over actions, with
`skip` as the reference category. The feature map `phi` is the theory of
engagement — swapping theories means swapping `phi` and its `theta`, nothing
else in the codebase changes.

    U_a(u, p) = theta_a^T phi(x_u, d_p, ctx)         a in {like, reply, repost, quote, report}
    U_skip    = 0
    P(a | u, p) = exp(U_a) / (1 + sum_a' exp(U_a'))
"""

from __future__ import annotations

import numpy as np

from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.exposure.attention import Exposures
from discourse_lab.population import Population
from discourse_lab.registry import get, names, register

ACTIONS = ("like", "reply", "repost", "quote", "report")

# Feature vocabulary (spec §2.6), plus `disagree_x_con` — the interaction the
# spec's own "critical sign" note requires: outrage needs a term whose
# effective weight is negative specifically for high-contrarianism users,
# which a kernel linear in a single global theta cannot express from
# `agreement` alone. `disagree_x_con = -agreement * contrarianism_u` is that
# term: a plain positive theta on it reproduces "disagreement raises
# engagement, more so the more contrarian the user".
FEATURES = (
    "intercept",
    "affinity",
    "agreement",
    "arousal",
    "arousal_x_neu",
    "provoc_x_con",
    "disagree_x_con",
    "prominence",
    "social_proof",
    "tie_strength",
    "quality",
    "novelty",
    "specificity",
    "recency",
    "credulity_x_q",
)

# Per-action intercepts — the baseline propensity to do anything at all.
#
# These are load-bearing and were missing until calibration. Without them
# every action has U_a = 0 before features are consulted, so
# P(skip) = 1/(1+5) = 17% and the simulation engages on 83% of exposures no
# matter what the kernel is: even `null`, which the spec describes as
# "intercept only", had no intercept. Real platforms sit at a few percent,
# and the difference propagates everywhere — an 80% engagement rate drove
# R_eff to ~16 against the spec's requirement of E[R] < 1, which in turn
# made 71% of cascades non-singletons against a >90% singleton target.
#
# Levels are set so a featureless exposure engages ~4% of the time with a
# plausible action mix (roughly 65% like, 14% repost, 14% reply, 5% quote,
# <1% report), and so the repost rate keeps R_eff subcritical at the default
# mean degree of 40.
# The repost/quote/reply levels are additionally set against spec §5.1's
# cascade shape (>90% of roots stay singletons). Reply *posts* are no longer
# produced here at all — they are scheduled by the Hawkes intensity in
# dynamics/hawkes.py per §2.3 — so the "reply" intercept now governs only the
# reply engagement *event*, which feeds the discourse-state update and drift
# channel 2.
DEFAULT_INTERCEPTS: dict[str, float] = {
    "like": -3.5,
    "repost": -7.0,
    "reply": -8.5,
    "quote": -8.0,
    "report": -8.0,
}

_INTERCEPT_ENTRIES = tuple(
    (action, "intercept", weight) for action, weight in DEFAULT_INTERCEPTS.items()
)



def _with_intercepts(entries: tuple[tuple[str, str, float], ...]) -> tuple[tuple[str, str, float], ...]:
    """Every kernel carries the baseline; features move a user off it."""
    return _INTERCEPT_ENTRIES + entries


# theta entries: (action, feature, weight). Dominant terms per dev §6 step 6
# table; unlisted (action, feature) pairs are zero.
KERNEL_THETAS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "homophily": _with_intercepts((
        ("like", "affinity", 1.0), ("like", "agreement", 1.0),
        ("repost", "affinity", 0.8), ("repost", "agreement", 0.8),
        ("reply", "affinity", 0.5), ("reply", "agreement", 0.5),
    )),
    "outrage": _with_intercepts((
        ("reply", "disagree_x_con", 1.5), ("reply", "arousal", 1.0),
        ("quote", "disagree_x_con", 1.3), ("quote", "arousal", 0.8),
        ("report", "disagree_x_con", 1.0),
    )),
    "bandwagon": _with_intercepts((
        ("like", "social_proof", 1.0), ("like", "prominence", 0.6),
        ("repost", "social_proof", 1.2), ("repost", "prominence", 0.8),
    )),
    "epistemic": _with_intercepts((
        ("like", "quality", 1.0), ("like", "novelty", 0.4),
        ("quote", "quality", 0.8), ("quote", "specificity", 0.4),
    )),
    # spec §2.6: "intercept only" — the pure structural baseline, with no
    # feature dependence at all
    "null": _with_intercepts(()),
}
for _name, _entries in KERNEL_THETAS.items():
    register("kernel_theta", _name)(_entries)


def compute_features(
    exposures: Exposures,
    posts: PostBatch,
    pop: Population,
    is_follower: np.ndarray,
    t_current: int,
) -> dict[str, np.ndarray]:
    names_ = pop.trait_names
    topic_cols = [i for i, n in enumerate(names_) if n.startswith("topic_affinity_")]
    stance_cols = [i for i, n in enumerate(names_) if n.startswith("stance_")]

    u = exposures.user_id
    p = exposures.post_idx
    author = posts.author[p]

    a_u = pop.X_used[u][:, topic_cols]
    topic_p = posts.topic[p]
    affinity = a_u[np.arange(len(u)), topic_p]

    s_u = pop.X_used[u][:, stance_cols]
    s_p = posts.stance[p]
    agreement = -np.linalg.norm(s_u - s_p, axis=1)

    neuroticism = pop.X_used[u, names_.index("neuroticism")]
    contrarianism = pop.X_used[u, names_.index("contrarianism")]
    credulity = pop.X_used[u, names_.index("credulity")]
    prominence_author = pop.X_used[author, names_.index("prominence")]

    arousal = posts.arousal[p]
    return {
        "intercept": np.ones(len(u)),
        "affinity": affinity,
        "agreement": agreement,
        "arousal": arousal,
        "arousal_x_neu": arousal * neuroticism,
        "provoc_x_con": posts.provocativeness[p] * contrarianism,
        "disagree_x_con": (-agreement) * contrarianism,
        "prominence": np.log1p(prominence_author),
        "social_proof": np.log1p(posts.engagement_count[p]),
        "tie_strength": is_follower.astype(float),
        "quality": posts.quality[p],
        "novelty": posts.novelty[p],
        "specificity": posts.specificity[p],
        "recency": -(t_current - posts.t[p]),
        "credulity_x_q": credulity * (1 - posts.specificity[p]),
    }


def kernel_names() -> list[str]:
    return names("kernel_theta")


def apply_kernel(
    theta_entries: tuple[tuple[str, str, float], ...],
    features: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one action per exposure (one of ACTIONS, or "skip")."""
    m = len(next(iter(features.values())))
    utility = {a: np.zeros(m) for a in ACTIONS}
    for action, feature, weight in theta_entries:
        utility[action] += weight * features[feature]

    exp_u = np.stack([np.exp(utility[a]) for a in ACTIONS], axis=1)  # (m, |ACTIONS|)
    denom = 1.0 + exp_u.sum(axis=1)
    probs = np.concatenate([(1.0 / denom)[:, None], exp_u / denom[:, None]], axis=1)  # skip first

    cdf = np.cumsum(probs, axis=1)
    u = rng.random((m, 1))
    choice = (u < cdf).argmax(axis=1)

    labels = np.array(("skip",) + ACTIONS)
    return labels[choice]


def named_kernel(name: str) -> tuple[tuple[str, str, float], ...]:
    return get("kernel_theta", name)
