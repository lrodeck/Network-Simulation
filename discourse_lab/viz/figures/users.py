"""The population up close: the follow network laid out in stance space, and
what each person's feed delivered against what existed.

Layout note. These are drawn in **stance space** (the first two principal
components of the population's stance matrix), not with a force-directed
layout. A spring layout is O(N^2), non-deterministic, and its axes mean
nothing --- two nodes end up adjacent because the solver put them there. Here
a node's position *is* its ideological position, so a visible cluster is a
claim you can check, and an edge that crosses the plot is a tie that crosses
the cleavage.
"""

from __future__ import annotations

from discourse_lab.viz._deps import require_matplotlib
from discourse_lab.viz.style import (
    CHROME,
    FIG_DOUBLE_COL,
    FIG_SINGLE_COL,
    PALETTE,
    diverging_cmap,
    series_colors,
    styled,
    symmetric_norm,
)

MAX_NODES = 250     # beyond this the edges are a solid block, not a network
MAX_EDGES = 700     # ditto, and a dense subgraph hits this well before MAX_NODES


def stance_layout(stance):
    """First two principal components of stance: coordinates, the variance each
    carries, and the loadings — a PC is a mix of axes, and which axis dominates
    it is something to read off the loading rather than assume."""
    import numpy as np

    centered = stance - stance.mean(axis=0)
    if centered.shape[1] == 1:
        coords = np.column_stack([centered[:, 0], np.zeros(len(centered))])
        return coords, (1.0, 0.0), np.array([[1.0], [0.0]])
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    var = (s ** 2) / (s ** 2).sum()
    return coords, (float(var[0]), float(var[1])), vt[:2]


def fig_user_network(graph, users, lex=None, color="camp", n_nodes=MAX_NODES,
                     max_edges=MAX_EDGES, seed=0,
                     title="Who follows whom, in stance space"):
    """The follow graph on a subsample, positioned by ideology.

    Nodes are sized by followers and coloured by camp (or `archetype`). The
    subsample is drawn with probability proportional to followers + 1 so the
    accounts that actually carry the attention are in the picture — a uniform
    sample of a heavy-tailed graph is a picture of lurkers.

    Edges are a **uniform random subsample of the induced subgraph**, coloured
    by whether they cross the camp boundary. Uniform, not one cap per class: a
    per-class cap would draw every cross-camp tie and only some same-camp ones,
    which makes a mixed network look segregated in exactly the way the eye is
    worst at correcting for. The measured crossing share goes in the title, so
    the number is there even when the picture is dense.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    stance_cols = [c for c in users.columns if c.startswith("stance_")]
    if not stance_cols:
        raise ValueError("user table carries no stance columns")
    stance = np.column_stack([users[c].to_numpy() for c in stance_cols])
    coords, var, loadings = stance_layout(stance)

    followers = (users["followers"].to_numpy().astype(float)
                 if "followers" in users.columns else np.zeros(len(stance)))
    rng = np.random.default_rng(seed)
    n_show = min(n_nodes, len(stance))
    weights = followers + 1.0
    idx = np.sort(rng.choice(len(stance), size=n_show, replace=False,
                             p=weights / weights.sum()))

    sub = graph.csr[idx][:, idx].tocoo()
    xy = coords[idx]
    camp = users["camp"].to_numpy()[idx]
    crossing = camp[sub.row] != camp[sub.col]

    n_edges = len(sub.row)
    rng2 = np.random.default_rng(seed + 1)
    shown = (np.arange(n_edges) if n_edges <= max_edges
             else rng2.choice(n_edges, size=max_edges, replace=False))
    cross_share = float(crossing.mean()) if n_edges else float("nan")

    with styled():
        fig, ax = plt.subplots(figsize=(FIG_DOUBLE_COL, FIG_DOUBLE_COL * 0.8))
        for mask, colour, width, alpha, z in (
            (~crossing[shown], CHROME["muted"], 0.4, 0.35, 1),
            (crossing[shown], PALETTE[2], 0.4, 0.5, 2),
        ):
            rows = shown[mask]
            if not len(rows):
                continue
            segments = np.stack([xy[sub.row[rows]], xy[sub.col[rows]]], axis=1)
            ax.add_collection(LineCollection(segments, colors=colour,
                                             linewidths=width, alpha=alpha, zorder=z))

        labels = users[color].to_numpy()[idx]
        levels = sorted(set(labels.tolist()))
        colours = series_colors(len(levels))
        sizes = 12 + 90 * (followers[idx] / max(followers.max(), 1.0))
        for level, colour in zip(levels, colours):
            mask = labels == level
            name = f"camp {level}" if color == "camp" else str(level)
            ax.scatter(xy[mask, 0], xy[mask, 1], s=sizes[mask], c=colour,
                       edgecolors="white", linewidths=0.4, zorder=3, label=name)

        names = [lex.axis_label(d) for d in range(len(stance_cols))] if lex is not None \
            else stance_cols
        lead = [names[int(np.argmax(np.abs(loadings[k])))] for k in (0, 1)]
        ax.set_xlabel(f"stance PC1 — {var[0]:.0%} of variance, loads mostly on {lead[0]}")
        ax.set_ylabel(f"stance PC2 — {var[1]:.0%}, mostly {lead[1]}")
        ax.set_title(
            f"{title}\n{n_show} of {len(stance)} users sampled by followers · "
            f"{len(shown)} of {n_edges} ties drawn · "
            f"{cross_share:.0%} of ties cross the camp line (highlighted)",
            fontsize="small")
        ax.legend(frameon=False, loc="best", fontsize="small",
                  title="node colour", title_fontsize="small")
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
    return fig


def fig_feed_mirror(feeds, users=None, by="archetype",
                    title="What the feed showed, against what existed"):
    """Per-user feed distance vs. world distance, with the y = x line.

    Everything below the diagonal is a feed closer to its viewer than the
    world is. The gap is the filter-bubble effect, in stance units, per person
    --- not a single population average that could hide both directions.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    frame = feeds if users is None else feeds.join(users, on="user", how="inner")
    world = frame["world_dist"].to_numpy()
    feed = frame["feed_dist"].to_numpy()
    ok = np.isfinite(world) & np.isfinite(feed)
    world, feed = world[ok], feed[ok]

    with styled():
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(FIG_DOUBLE_COL, FIG_SINGLE_COL * 0.95),
            gridspec_kw={"width_ratios": (1.25, 1.0)})

        if users is not None and by in frame.columns:
            labels = frame[by].to_numpy()[ok]
            levels = sorted(set(labels.tolist()))
            for level, colour in zip(levels, series_colors(len(levels))):
                mask = labels == level
                ax.scatter(world[mask], feed[mask], s=9, c=colour, alpha=0.65,
                           linewidths=0, label=str(level))
            ax.legend(frameon=False, fontsize="x-small", title=by,
                      title_fontsize="x-small", loc="upper left")
        else:
            ax.scatter(world, feed, s=9, c=PALETTE[0], alpha=0.6, linewidths=0)

        lo = float(min(world.min(), feed.min()))
        hi = float(max(world.max(), feed.max()))
        ax.plot([lo, hi], [lo, hi], color=CHROME["ink"], lw=0.8, ls="--", zorder=0)
        ax.annotate("feed = world", xy=(hi, hi), xytext=(-4, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize="x-small", color=CHROME["muted"])
        ax.set_xlabel("distance to posts that existed")
        ax.set_ylabel("distance to posts they saw")
        ax.set_title("per user")

        bubble = feed - world
        ax2.hist(bubble, bins=40, color=PALETTE[0], edgecolor="white", linewidth=0.3)
        ax2.axvline(0, color=CHROME["ink"], lw=0.8, ls="--")
        ax2.axvline(float(np.mean(bubble)), color=PALETTE[2], lw=1.4)
        ax2.annotate(f"mean {np.mean(bubble):+.3f}",
                     xy=(float(np.mean(bubble)), ax2.get_ylim()[1]),
                     xytext=(4, -8), textcoords="offset points",
                     fontsize="x-small", color=PALETTE[2], va="top")
        ax2.set_xlabel("feed - world  (negative = closer than the world)")
        ax2.set_ylabel("users")
        ax2.set_title("the gap")

        fig.suptitle(title)
    return fig
