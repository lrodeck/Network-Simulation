"""Experiment 02 (DMP regime mapping) §1-2: cascade-level (Digital Micro
Public) outcomes, computed from a run's persisted `posts`/`exposures`/
`traits` tables.

A DMP is one cascade: a root post plus the transitive closure of replies to
it (`posts.root`). Two membership definitions are computed side by side,
per §1's requirement that neither be assumed without the other:

- **enacted**: users who posted anywhere in the cascade (`posts.author`).
- **audience**: users exposed to any post in the cascade (`exposures.user`
  joined on `exposures.post -> posts.root`), regardless of `attended`.

Per-member stance/animus movement is read off the nearest available trait
snapshots bracketing the cascade's own [t_start, t_end] window, not global
run-start/run-end values — a cascade active at ticks 200-210 needs trait
readings near 200 and 210, not a snapshot 190+ ticks (and every other
cascade's activity) away. This means the run must have been persisted with
snapshots dense enough that most cascades' windows are bracketed by DISTINCT
before/after ticks; a run with only two snapshots (t=0, t=T-1) cannot
support this analysis at all, and `dmp_outcomes` says so rather than
silently returning global deltas mislabeled as cascade-level ones.
"""

from __future__ import annotations

import numpy as np
import polars as pl

MIN_CASCADE_SIZE = 2  # singletons are publications, not DMPs (spec doc §1)


def cascade_windows(posts: pl.DataFrame) -> pl.DataFrame:
    """Per-root (t_start, t_end, size, max_depth), one row per cascade."""
    return posts.group_by("root").agg(
        pl.col("t").min().alias("t_start"),
        pl.col("t").max().alias("t_end"),
        pl.len().alias("size"),
        pl.col("depth").max().alias("max_depth"),
    )


def cascade_singleton_mask(windows: pl.DataFrame) -> pl.Series:
    return windows["size"] < MIN_CASCADE_SIZE


def enacted_members(posts: pl.DataFrame) -> pl.DataFrame:
    """(root, user) pairs: users who posted in that cascade."""
    return posts.select(["root", "author"]).unique().rename({"author": "user"})


def audience_members(exposures: pl.DataFrame, posts: pl.DataFrame) -> pl.DataFrame:
    """(root, user) pairs: users exposed to any post in that cascade.

    "Exposed" per §1's own definition -- not filtered by `attended`, which
    is a separate, narrower construct already used elsewhere
    (`selection_filtering`).
    """
    post_root = posts.select(pl.col("id"), pl.col("root"))
    return (
        exposures.join(post_root, left_on="post", right_on="id", how="inner")
        .select(["root", "user"])
        .unique()
    )


def _bracket_snapshot_ticks(windows: pl.DataFrame, snap_ticks: np.ndarray) -> pl.DataFrame:
    """Attach `before_t`/`after_t`: the tightest available snapshot ticks
    with `before_t <= t_start` and `after_t >= t_end`. NaN (as -1 sentinel,
    filtered by caller) when the cascade's window isn't bracketed at all
    (e.g. it starts before the first snapshot, or the run ended before the
    cascade closed and there is no later snapshot).
    """
    snap_ticks = np.sort(np.asarray(snap_ticks))
    t_start = windows["t_start"].to_numpy()
    t_end = windows["t_end"].to_numpy()
    before_idx = np.searchsorted(snap_ticks, t_start, side="right") - 1
    after_idx = np.searchsorted(snap_ticks, t_end, side="left")
    before_t = np.where(before_idx >= 0, snap_ticks[np.clip(before_idx, 0, len(snap_ticks) - 1)], -1)
    after_t = np.where(
        after_idx < len(snap_ticks), snap_ticks[np.clip(after_idx, 0, len(snap_ticks) - 1)], -1
    )
    return windows.with_columns(
        pl.Series("before_t", before_t), pl.Series("after_t", after_t)
    )


def dmp_outcomes(
    posts: pl.DataFrame,
    exposures: pl.DataFrame,
    traits: pl.DataFrame,
    membership: str = "enacted",
    max_bracket_to_lifespan_ratio: float = 5.0,
) -> pl.DataFrame:
    """Per-cascade §2 outcomes under one membership definition.

    `traits` must be `RunHandle.traits_used(cfg)` -- real units, not the
    stored link space. Requires `animus` and at least one `stance_*` column.

    Returns one row per non-singleton cascade with a usable snapshot
    bracket: `root`, `size`, `max_depth`, `lifespan` (`t_end - t_start`),
    `n_members`, `delta_stance` (signed, first stance axis), `delta_animus`,
    plus `before_t`/`after_t` for auditing how tight the bracket was.

    Raises `ValueError` if the median snapshot-bracket width
    (`after_t - before_t`) across non-singleton cascades exceeds
    `max_bracket_to_lifespan_ratio` times their own lifespan
    (`t_end - t_start`) -- the signal this run's snapshot density is too
    coarse to support cascade-level attribution at all. A run with only two
    global snapshots (t=0, t=T-1) gives every cascade a "distinct" bracket
    trivially (before != after) without this check, which is why distinctness
    alone is not the right guard: what matters is whether the bracket is
    tight around the cascade's OWN window, not merely non-degenerate.
    Silently returning global-window deltas mislabeled as cascade-level ones
    would be a worse failure mode than refusing.
    """
    if membership not in ("enacted", "audience"):
        raise ValueError(f"membership must be 'enacted' or 'audience', got {membership!r}")
    stance_cols = [c for c in traits.columns if c.startswith("stance_")]
    if not stance_cols or "animus" not in traits.columns:
        raise ValueError(f"traits frame lacks stance_*/animus columns; has {sorted(traits.columns)}")
    stance_col = stance_cols[0]

    windows = cascade_windows(posts)
    windows = windows.filter(pl.col("size") >= MIN_CASCADE_SIZE)
    if windows.height == 0:
        return windows.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("delta_stance"),
            pl.lit(None, dtype=pl.Float64).alias("delta_animus"),
        )

    snap_ticks = traits["t"].unique().to_numpy()
    windows = _bracket_snapshot_ticks(windows, snap_ticks)
    windows = windows.filter((pl.col("before_t") >= 0) & (pl.col("after_t") >= 0))
    windows = windows.filter(pl.col("after_t") > pl.col("before_t"))

    if windows.height == 0:
        raise ValueError(
            "no non-singleton cascade has a usable snapshot bracket at all -- this run's "
            "trait-snapshot density is too coarse for cascade-level attribution."
        )
    bracket_width = (windows["after_t"] - windows["before_t"]).to_numpy()
    lifespan = np.maximum((windows["t_end"] - windows["t_start"]).to_numpy(), 1)
    ratio = np.median(bracket_width / lifespan)
    if ratio > max_bracket_to_lifespan_ratio:
        raise ValueError(
            f"median snapshot-bracket width is {ratio:.1f}x the median cascade's own lifespan "
            f"(threshold {max_bracket_to_lifespan_ratio}x) -- this run's trait-snapshot density is "
            "too coarse for cascade-level attribution (a bracket many times wider than the cascade "
            "it's supposed to measure picks up every OTHER cascade's activity in between, not this "
            "one's). Re-run with a smaller dynamics.snapshot_every. A run with only two global "
            "snapshots (t=0, t=T-1) always fails this check, however 'distinct' before/after look."
        )

    members = enacted_members(posts) if membership == "enacted" else audience_members(exposures, posts)
    members = members.join(windows.select("root"), on="root", how="inner")

    before = traits.select(["t", "user", stance_col, "animus"]).rename(
        {stance_col: "stance_before", "animus": "animus_before"}
    )
    after = traits.select(["t", "user", stance_col, "animus"]).rename(
        {stance_col: "stance_after", "animus": "animus_after"}
    )

    tagged = members.join(
        windows.select(["root", "before_t", "after_t"]), on="root", how="inner"
    )
    tagged = tagged.join(before, left_on=["before_t", "user"], right_on=["t", "user"], how="inner")
    tagged = tagged.join(after, left_on=["after_t", "user"], right_on=["t", "user"], how="inner")
    tagged = tagged.with_columns(
        (pl.col("stance_after") - pl.col("stance_before")).alias("delta_stance_i"),
        (pl.col("animus_after") - pl.col("animus_before")).alias("delta_animus_i"),
    )

    per_cascade = tagged.group_by("root", maintain_order=True).agg(
        pl.len().alias("n_members"),
        pl.col("delta_stance_i").mean().alias("delta_stance"),
        pl.col("delta_animus_i").mean().alias("delta_animus"),
    )
    return windows.join(per_cascade, on="root", how="inner").with_columns(
        (pl.col("t_end") - pl.col("t_start")).alias("lifespan")
    )
