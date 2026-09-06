"""Who the users are, who they are connected to, and what they are shown.

Everything else in this library aggregates: a Gini, a camp share, a lever
effect. Those are the results, but they are also unfalsifiable-looking if you
cannot open one up and see a user. This module is the per-user view --- one
row per person, with their traits, their position in the network, and the
content their feed actually delivered.

The comparison that matters is the last one. A feed is only a filter bubble
relative to something; the honest baseline is **what existed at the same time**
--- every post published in the ticks a user was exposed in. `bubble` is
distance-to-my-feed minus distance-to-the-world, so it is negative when the
feed is closer to the viewer than the world is, and it is measured, not
assumed.

Deliberately *not* the candidate set the ranker scored. That set is not
persisted (spec 3.5 keeps a sampled exposure log, not a feed log) and
reconstructing it would mean re-running the tick. "What you saw vs. what
existed" is a weaker counterfactual than "what you saw vs. what you were
offered", and saying so is better than quietly substituting one for the other.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from discourse_lab.metrics.stylized import stance_clusters
from discourse_lab.semantics.lexicon import Lexicon


def _stance_matrix(df: pl.DataFrame, prefix: str, n: int) -> np.ndarray:
    return np.column_stack([df[f"{prefix}{d}"].to_numpy() for d in range(n)])


def user_table(handle, pop, graph=None, lex: Lexicon | None = None) -> pl.DataFrame:
    """One row per user: identity, position, and what the run did to them.

    Columns are named through the Lexicon where the model has names for them,
    so `stance_provision` reads as a cleavage rather than as `stance_0`.
    """
    lex = lex or Lexicon.from_handle(handle)
    n_users = pop.X_used.shape[0]
    traits = {name: pop.X_used[:, i] for i, name in enumerate(pop.trait_names)}
    stance = np.column_stack([traits[lex.trait_column(d)] for d in range(lex.n_axes)])
    camp = stance_clusters(stance)

    out: dict[str, np.ndarray | list] = {
        "user": np.arange(n_users),
        "archetype": [pop.archetype_names[i] for i in pop.archetype_labels],
        "camp": camp,
    }
    for d in range(lex.n_axes):
        out[lex.trait_column(d)] = stance[:, d]
    for name in ("activity", "prominence", "reply_prop", "contrarianism", "credulity"):
        if name in traits:
            out[name] = traits[name]

    if graph is not None:
        out["follows"] = np.asarray(graph.csr.sum(axis=1)).ravel().astype(np.int64)
        out["followers"] = np.asarray(graph.csr.sum(axis=0)).ravel().astype(np.int64)

    frame = pl.DataFrame(out)

    if handle.has_posts:
        posts = handle.posts()
        roots = posts.filter(pl.col("kind") == "post")
        frame = frame.join(
            roots.group_by("author").agg(pl.len().alias("posts")).rename({"author": "user"}),
            on="user", how="left")
        frame = frame.join(
            posts.filter(pl.col("kind") != "post")
                 .group_by("author").agg(pl.len().alias("replies")).rename({"author": "user"}),
            on="user", how="left")
        frame = frame.join(
            posts.group_by("author").agg(pl.col("engagement_count").sum().alias("attention"))
                 .rename({"author": "user"}),
            on="user", how="left")
        frame = frame.with_columns(
            [pl.col(c).fill_null(0) for c in ("posts", "replies", "attention")])

    if handle.has_exposures:
        exposures = handle.exposures()
        frame = frame.join(
            exposures.group_by("user").agg(
                pl.len().alias("seen"),
                pl.col("is_follower").mean().alias("follower_share"),
                (pl.col("action") != "skip").mean().alias("engaged_share"),
            ), on="user", how="left")

    return frame.sort("user")


def feed_composition(handle, pop, lex: Lexicon | None = None) -> pl.DataFrame:
    """Per user: how far their feed sat from them, against what existed.

    Returns one row per user with `seen`, `feed_dist` (mean stance distance
    from the viewer to what they were shown), `world_dist` (the same distance
    to every post published in those same ticks), `bubble = feed_dist -
    world_dist`, plus the feed's topic entropy against the world's.

    Only users with at least `min_seen` sampled exposures are returned:
    `exposure_sample_rate` defaults to 1%, and a user with three sampled rows
    has a feed distance that is noise. Raise the rate rather than reading a
    thin table.
    """
    lex = lex or Lexicon.from_handle(handle)
    if not (handle.has_exposures and handle.has_posts):
        raise ValueError(
            "feed_composition needs both 'exposures' and 'posts' persisted; "
            "run with persist=('posts', 'engagements', 'exposures')."
        )

    D = lex.n_axes
    posts = handle.posts().select(
        ["id", "t", "topic"] + [f"stance_{d}" for d in range(D)]).rename({"id": "post"})
    exposures = handle.exposures().select(["t", "user", "post"])

    seen = exposures.drop("t").join(posts, on="post", how="inner")
    if seen.height == 0:
        return pl.DataFrame(schema={"user": pl.Int64})

    own = np.column_stack(
        [pop.X_used[:, pop.trait_names.index(lex.trait_column(d))] for d in range(D)])

    users = seen["user"].to_numpy()
    consumed = _stance_matrix(seen, "stance_", D)
    feed_d = np.linalg.norm(consumed - own[users], axis=1)

    # The baseline. For each thing a user saw, draw a post published in the
    # *same tick* uniformly at random and measure the distance to that instead:
    # same vintage, chosen by a coin flip rather than by the ranker. Matching on
    # the consumed post's tick (not the exposure tick) is what makes the pool
    # guaranteed non-empty -- it contains the real post by construction.
    rng = np.random.default_rng(0)
    all_stance = _stance_matrix(posts, "stance_", D)
    all_topic = posts["topic"].to_numpy()
    post_t = posts["t"].to_numpy()

    order = np.argsort(post_t, kind="stable")
    sorted_t = post_t[order]
    seen_t = seen["t"].to_numpy()
    lo = np.searchsorted(sorted_t, seen_t, side="left")
    hi = np.searchsorted(sorted_t, seen_t, side="right")
    world_rows = order[lo + (rng.random(len(seen_t)) * (hi - lo)).astype(np.int64)]

    world_d = np.linalg.norm(all_stance[world_rows] - own[users], axis=1)

    n_topics = int(all_topic.max()) + 1
    n_users = own.shape[0]

    # per-user topic histograms, both feeds at once. A polars group-by with a
    # python UDF per group was the first version and it is ~40x slower here for
    # a computation that is two scatter-adds.
    feed_topic = seen["topic"].to_numpy()
    world_topic = all_topic[world_rows]
    feed_hist = np.zeros((n_users, n_topics))
    world_hist = np.zeros((n_users, n_topics))
    np.add.at(feed_hist, (users, feed_topic), 1.0)
    np.add.at(world_hist, (users, world_topic), 1.0)

    present = np.unique(users)
    frame = pl.DataFrame({
        "user": users, "feed_dist": feed_d, "world_dist": world_d,
    }).group_by("user").agg(
        pl.len().alias("seen"),
        pl.col("feed_dist").mean(),
        pl.col("world_dist").mean(),
    )

    def _row_entropy(hist: np.ndarray) -> np.ndarray:
        totals = hist.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(totals > 0, hist / totals, 0.0)
            terms = np.where(p > 0, -p * np.log(p), 0.0)
        return np.where(totals.ravel() > 0, terms.sum(axis=1), np.nan)

    extra = pl.DataFrame({
        "user": present,
        "feed_topic_entropy": _row_entropy(feed_hist)[present],
        "world_topic_entropy": _row_entropy(world_hist)[present],
        "top_topic": feed_hist[present].argmax(axis=1),
    })
    return frame.join(extra, on="user", how="left").with_columns(
        (pl.col("feed_dist") - pl.col("world_dist")).alias("bubble"),
        pl.col("top_topic").map_elements(lex.topic_label, return_dtype=pl.Utf8)
          .alias("top_topic_name"),
    ).sort("user")


def audience_summary(users: pl.DataFrame, feeds: pl.DataFrame, by: str = "archetype",
                     min_seen: int = 10) -> pl.DataFrame:
    """`user_table` x `feed_composition`, grouped.

    `min_seen` drops users whose sampled feed is too thin to average. Reported
    as a column so a group that vanished is visible rather than silent.
    """
    joined = users.join(feeds, on="user", how="inner")
    kept = joined.filter(pl.col("seen") >= min_seen)
    return kept.group_by(by).agg(
        pl.len().alias("n"),
        pl.col("seen").mean().alias("seen"),
        pl.col("posts").mean().alias("posts") if "posts" in kept.columns else pl.lit(None).alias("posts"),
        pl.col("attention").mean().alias("attention") if "attention" in kept.columns else pl.lit(None).alias("attention"),
        pl.col("feed_dist").mean(),
        pl.col("world_dist").mean(),
        pl.col("bubble").mean(),
        pl.col("feed_topic_entropy").mean().alias("feed_entropy"),
    ).sort(by)
