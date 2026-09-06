"""Normative outcomes: the constructs a democratic-discourse study reports.

spec §5.1's stylized facts are a *validity gate* — does this behave enough like
a platform to reason from. These are the dependent variables: what changes when
a platform designer changes something.

Each is named as a construct rather than a metric, because naming it commits to
an interpretation and that commitment should be visible and arguable. The
operationalisation and its weaknesses are in each docstring.

Scope limit worth stating before any of these are read as more than they are:
the model has no deliberation, no argument and no persuasion-by-reason. Drift
is exposure-weighted social influence (spec §2.9). So these speak to structural
preconditions for democratic discourse — who is heard, who meets disagreement,
whether attention tracks merit — and not to the quality of deliberation itself.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from discourse_lab.registry import get, names, register


def _camp_labels(pop, lex):
    from discourse_lab.metrics.stylized import stance_clusters

    return stance_clusters(pop.X_used[:, lex.stance_columns(pop.trait_names)])


def _join_authors(frame: pl.DataFrame, posts: pl.DataFrame) -> pl.DataFrame:
    return frame.join(posts.select(["id", "author"]), left_on="post", right_on="id", how="inner")


@register("outcome", "cross_cutting_exposure")
def cross_cutting_exposure(handle, pop, lex, delta: float | None = None,
                           top_ranks: int = 5) -> dict:
    """Does the platform put people in front of the other side?

    The most established construct in the democratic-discourse literature
    (exposure to disagreement), and the reason the §3.5 1% exposure sample
    exists. Two complementary readings, both reported:

    `camp_share` — the fraction of exposures whose author is in the other camp,
    against a chance baseline of 0.50 for two equal camps. Coarse but directly
    interpretable.

    `stance_share` — 1 - `metrics.echo_chamber_index`, the fraction of consumed
    stance mass *outside* `delta` of the viewer's own position. Continuous, and
    the spec §5.2 measure.

    `delta` defaults to **half the median distance between two random users**,
    so "agrees with me" means "closer than half a typical disagreement" at any
    dimensionality. A constant would not survive a change of D, and neither
    does the obvious fix of scaling by the per-axis standard deviation: in D
    dimensions a typical distance is ~sqrt(D) times that, so `0.5 * std` put
    delta at 0.25 against typical distances near 0.87 and classified 94% of
    everything as cross-cutting — a number that moves with nothing.

    `algorithmic_share` isolates the exposures the viewer did *not* subscribe to
    (`is_follower == False`), which is the part a recommender actually controls.
    Measured, injected items are far more cross-cutting than the feed they land
    in (0.396 vs 0.288) while barely moving the aggregate — the mechanism works
    per item and is drowned out by follower fanout at any inject_k a platform
    would plausibly ship. It is NaN whenever `inject_k == 0`: random injection
    is the *only* source of non-follower candidates, so the subset is empty and
    no ranker setting can make it non-empty. That is a property of the feed
    construction, not a measurement failure -- a lever sweep over rankers at
    inject_k=0 will report this column as all-NaN by construction.

    `rank_penalty` is the measure that survives that. It is the cross-cutting
    share of the items the ranker put **at the top of the feed** minus the
    share of everything below, using the persisted `rank`. That is the
    recommender's actual contribution: the candidate pool is fixed by the graph
    and by injection, and all a ranker does is order it, so a ranker that
    demotes disagreement shows up here as a negative number. Unlike
    `algorithmic_share` it is defined at every `inject_k`, including zero,
    because ordering exists whether or not anything was injected. Zero under
    `chronological`, which orders by recency and cannot see stance.

    `top_ranks` sets the cut. It is a rank count and not a quantile: ranks are
    already truncated by position decay (`tau_position`), so a quantile of the
    surviving rows would move with the truncation rather than with the ranker.

    Sampling caveat: at `exposure_sample_rate=0.01` a user contributes ~2 rows
    per run, so only the population mean is interpretable, never a per-user
    value.
    """
    from discourse_lab.metrics import echo_chamber_index

    exposures, posts = handle.exposures(), handle.posts()
    joined = _join_authors(exposures, posts)
    if len(joined) == 0:
        return {"camp_share": float("nan"), "stance_share": float("nan"),
                "algorithmic_share": float("nan"), "rank_penalty": float("nan"),
                "n": 0}

    labels = _camp_labels(pop, lex)
    users = joined["user"].to_numpy()
    authors = joined["author"].to_numpy()
    crossing = labels[users] != labels[authors]

    is_follower = joined["is_follower"].to_numpy()
    injected = ~is_follower

    stance_cols = [lex.post_column(d) for d in range(lex.n_axes)]
    consumed = (
        exposures.join(posts.select(["id", *stance_cols]), left_on="post", right_on="id", how="inner")
    )
    own = pop.X_used[:, lex.stance_columns(pop.trait_names)]
    if delta is None:
        rng = np.random.default_rng(0)
        n = own.shape[0]
        sample = min(20_000, n * 4)
        a, b = rng.integers(0, n, sample), rng.integers(0, n, sample)
        delta = 0.5 * float(np.median(np.linalg.norm(own[a] - own[b], axis=1)))
    index = echo_chamber_index(
        own,
        consumed.select(stance_cols).to_numpy(),
        consumed["user"].to_numpy(),
        delta=delta,
    )

    rank = joined["rank"].to_numpy()
    top = rank < top_ranks
    rank_penalty = (float(crossing[top].mean() - crossing[~top].mean())
                    if top.any() and (~top).any() else float("nan"))

    return {
        "camp_share": float(crossing.mean()),
        "stance_share": float(1.0 - np.nanmean(index)),
        "algorithmic_share": float(crossing[injected].mean()) if injected.any() else float("nan"),
        "rank_penalty": rank_penalty,
        "delta": float(delta),
        "n": int(len(joined)),
    }


@register("outcome", "voice_inequality")
def voice_inequality(handle, pop, lex) -> dict:
    """Who gets to speak, and who gets heard.

    Two Ginis (posting volume, attention received) plus the share of attention
    reaching the smaller camp. The aggregate Gini can look healthy while one
    camp is inaudible, and "is the minority heard at all" is the question a
    democratic-discourse study is actually asking.
    """
    from discourse_lab.measures import gini
    from discourse_lab.metrics import attention_inequality
    from discourse_lab.metrics.stylized import posting_volume_gini

    posts = handle.posts()
    roots = posts.filter(posts["kind"] == "post")
    n_users = pop.X_used.shape[0]
    engagement = posts["engagement_count"].to_numpy().astype(float)

    attention_gini, top1 = attention_inequality(engagement)
    labels = _camp_labels(pop, lex)
    authors = posts["author"].to_numpy()
    by_camp = np.array([engagement[labels[authors] == c].sum() for c in (0, 1)])
    smaller = int(np.argmin([(labels == 0).sum(), (labels == 1).sum()]))
    minority_share = float(by_camp[smaller] / by_camp.sum()) if by_camp.sum() > 0 else float("nan")

    return {
        "posting_gini": float(posting_volume_gini(roots["author"].to_numpy(), n_users)),
        "attention_gini": float(attention_gini),
        "top1_share": float(top1),
        "minority_camp_attention_share": minority_share,
    }


@register("outcome", "epistemic_alignment")
def epistemic_alignment(handle, pop=None, lex=None) -> dict:
    """Does attention track merit?

    Spearman(quality, engagement) over posts. spec §1.2 says `quality` exists
    "specifically so you can ask whether your engagement kernel correlates
    attention with merit".

    **The weakest of the four, and it must be read as a difference against the
    matched null, never as a level.** `quality` is generated by the expression
    map from author traits, topic and the discourse state (spec §2.4), so it is
    correlated with author identity by construction; the level therefore partly
    measures the data-generating process rather than the platform. Only the
    model-minus-null contrast isolates what the feed and kernel did.
    """
    from discourse_lab.metrics import quality_attention_correlation

    posts = handle.posts()
    return {
        "quality_attention_rho": float(
            quality_attention_correlation(
                posts["quality"].to_numpy(), posts["engagement_count"].to_numpy()
            )
        )
    }


@register("outcome", "hostility_given_contact")
def hostility_given_contact(handle, pop, lex) -> dict:
    """When people do meet across the divide, how badly does it go?

    Returns the contact rate and the hostility ratio **together**, and they must
    be read together: a platform that eliminates cross-camp contact entirely
    scores a perfect zero on hostility while being the opposite of what a
    democratic-discourse intervention is trying to achieve.
    """
    from discourse_lab.metrics.stylized import inter_cluster_interaction

    engagements, posts = handle.engagements(), handle.posts()
    joined = _join_authors(engagements, posts)
    if len(joined) == 0:
        return {"contact_rate": float("nan"), "hostility": float("nan"), "n": 0}

    labels = _camp_labels(pop, lex)
    rate, hostility = inter_cluster_interaction(
        joined["user"].to_numpy(), joined["author"].to_numpy(),
        joined["action"].to_numpy(), labels,
    )
    return {"contact_rate": float(rate), "hostility": float(hostility), "n": int(len(joined))}


@register("outcome", "feed_narrowing")
def feed_narrowing(handle, pop, lex) -> dict:
    """How much narrower the feed was than the world it was drawn from.

    The other four constructs describe what the platform delivered. This one
    describes what it *withheld*, by measuring each exposure against a post
    published in the same tick chosen at random --- same vintage, ordered by a
    coin flip instead of by the ranker. See `discourse_lab.users` for why the
    baseline is "what existed" rather than "what the ranker was offered".

    `bubble` is negative when feeds sit closer to their viewer than the world
    does, in stance units. `topic_narrowing` is the same idea for variety: the
    feed's Shannon entropy over topics minus the world's.
    """
    from discourse_lab.users import feed_composition

    feeds = feed_composition(handle, pop, lex)
    if feeds.height == 0:
        return {"bubble": float("nan"), "topic_narrowing": float("nan"), "n_users": 0}
    return {
        "feed_dist": float(feeds["feed_dist"].mean()),
        "world_dist": float(feeds["world_dist"].mean()),
        "bubble": float(feeds["bubble"].mean()),
        "topic_narrowing": float(feeds["feed_topic_entropy"].mean()
                                 - feeds["world_topic_entropy"].mean()),
        "n_users": int(feeds.height),
    }


def outcome_names() -> list[str]:
    return names("outcome")


def normative_outcomes(handle, pop=None, lex=None) -> dict[str, float]:
    """Every outcome the run has the data for, flattened to `name.field`.

    Degrades the same way `stylized_facts_from_run` does: a run persisted
    without exposures simply omits the constructs that need them, rather than
    failing or silently reporting a wrong number.
    """
    from discourse_lab.semantics import Lexicon

    lex = lex if lex is not None else Lexicon.from_handle(handle)
    out: dict[str, float] = {}

    needs = {
        "cross_cutting_exposure": ("posts", "exposures"),
        "voice_inequality": ("posts",),
        "epistemic_alignment": ("posts",),
        "hostility_given_contact": ("posts", "engagements"),
        "feed_narrowing": ("posts", "exposures"),
    }
    for name in outcome_names():
        if not all(getattr(handle, f"has_{table}") for table in needs[name]):
            continue
        result = get("outcome", name)(handle, pop, lex)
        for key, value in result.items():
            out[f"{name}.{key}"] = value
    return out
