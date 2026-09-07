"""Intervention results: what each design lever does to each normative outcome."""

from __future__ import annotations

from discourse_lab.viz._deps import require_matplotlib
from discourse_lab.viz.style import CHROME, FIG_DOUBLE_COL, STATUS, series_colors, styled


def fig_lever_effects(summary, outcomes=None, title="Design lever effects"):
    """Dot-and-interval per lever value, one panel per outcome, zero baseline.

    Plots `lever_effect` — the change from each lever's reference setting —
    not `kernel_delta`. The matched null holds the lever fixed, so its delta
    answers "was this mediated by the engagement kernel", which is a different
    question and roughly zero for feed levers by construction.

    Points, not bars: these are signed differences from a reference, and a bar
    implies magnitude from a meaningful zero. Values that do not clear twice
    their standard error are drawn hollow and labelled, so an unresolved effect
    cannot be read as a small one.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl

    frame = summary.filter(pl.col("lever_effect") != 0)
    if outcomes is not None:
        frame = frame.filter(pl.col("outcome").is_in(list(outcomes)))
    names = list(frame["outcome"].unique(maintain_order=True))
    colours = series_colors(min(len(names), 8))

    with styled():
        fig, axes = plt.subplots(
            1, len(names), figsize=(FIG_DOUBLE_COL, 0.42 * max(1, len(frame) // max(1, len(names))) + 1.6),
            sharey=True, squeeze=False,
        )
        for ax, outcome, colour in zip(axes[0], names, colours):
            part = frame.filter(pl.col("outcome") == outcome)
            labels = [f"{lever.split('.')[-1]}={value}"
                      for lever, value in zip(part["lever"], part["value"])]
            y = np.arange(len(labels))
            effect = part["lever_effect"].to_numpy().astype(float)
            err = 2 * np.nan_to_num(part["model_sd"].to_numpy().astype(float)) / np.sqrt(
                part["n_seeds"].to_numpy().astype(float)
            )
            resolves = part["resolves"].to_numpy()

            ax.axvline(0.0, color=CHROME["baseline"], linewidth=1.0)
            for i, solid in enumerate(resolves):
                ax.errorbar(
                    effect[i], y[i], xerr=err[i], fmt="o", color=colour, markersize=5,
                    elinewidth=1.4, capsize=0,
                    markerfacecolor=colour if solid else CHROME["surface"],
                    markeredgecolor=colour, markeredgewidth=1.2,
                )
            ax.set_title(outcome.split(".")[-1].replace("_", " "), loc="left", fontsize=8)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7)
        # row count from the frame, not from whatever `labels` the last panel
        # happened to leave behind
        n_rows = max(1, len(frame) // max(1, len(names)))
        axes[0][0].set_ylim(-0.6, n_rows - 0.4)
        fig.suptitle(f"{title}  (hollow = does not clear 2 s.e.)", x=0.01, ha="left", fontsize=9)
    return fig


def fig_contact_vs_hostility(summary, title="Contact and hostility"):
    """The trade-off that makes a single number misleading.

    A platform that eliminates cross-camp contact scores a perfect zero on
    hostility while being the opposite of what the intervention is for, so the
    two outcomes are plotted against each other and never reported alone. The
    desirable corner is high contact, low hostility — top left.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl

    contact = summary.filter(pl.col("outcome") == "hostility_given_contact.contact_rate")
    hostile = summary.filter(pl.col("outcome") == "hostility_given_contact.hostility")
    joined = contact.join(hostile, on=["lever", "value"], suffix="_h")

    # Each lever contributes its own reference row, so the baseline config
    # appears once per lever at exactly the same coordinates. Left alone those
    # labels overprint each other into an unreadable smear; merged, one point
    # carries every name that produced it.
    merged: dict[tuple[float, float], list[str]] = {}
    for lever, value, x, y in zip(
        joined["lever"], joined["value"],
        joined["model_mean"].to_numpy().astype(float),
        joined["model_mean_h"].to_numpy().astype(float),
    ):
        merged.setdefault((round(float(x), 6), round(float(y), 6)), []).append(
            f"{lever.split('.')[-1]}={value}"
        )
    (colour,) = series_colors(1)

    with styled():
        fig, ax = plt.subplots(figsize=(FIG_DOUBLE_COL / 2, 2.9))
        points = sorted(merged.items())
        x = np.array([p[0][0] for p in points])
        y = np.array([p[0][1] for p in points])
        ax.plot(x, y, "o", color=colour, markersize=6,
                markeredgecolor=CHROME["surface"], markeredgewidth=1.0)
        # Merging identical points is not enough — distinct-but-near points
        # still overprint. With a handful of conditions, alternating the label
        # above and below the marker separates them without a layout library.
        order = np.argsort(y)
        for rank, idx in enumerate(order):
            (xi, yi), names_here = points[idx]
            above = rank % 2 == 0
            ax.annotate(
                "\n".join(sorted(names_here)), (xi, yi), fontsize=6.5,
                xytext=(6, 6 if above else -6), textcoords="offset points",
                va="bottom" if above else "top", color=CHROME["secondary"],
            )
        ax.set_xlabel("cross-camp contact rate")
        ax.set_ylabel("hostility given contact")
        # the note goes in the subtitle, not in the axes: every corner of the
        # plot is somewhere a point can land
        ax.set_title(title, loc="left")
        fig.suptitle("desirable: more contact, less hostility",
                     x=0.01, ha="left", fontsize=6.5, color=CHROME["muted"])
        ax.margins(x=0.30, y=0.22)
    return fig


def fig_interaction(table, outcome_label="", top=8,
                    title="Does this lever do the same thing everywhere?"):
    """One row per (lever, value), one marker per background, joined by a line.

    Takes the frame `experiments.interaction_table` returns. A row whose
    markers sit on top of each other is a lever whose screen result was the
    whole story; a row whose markers are far apart is a lever whose main effect
    is an average over settings in which it does different things — and the
    line makes that length, which is `swing`, the thing the eye reads first.

    Sorted by swing and truncated to `top` rows, because the interesting claim
    is which levers interact, not an exhaustive listing of the ones that do not.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    background_cols = [c for c in table.columns if c not in ("lever", "value", "swing")]
    if len(background_cols) < 2:
        raise ValueError(
            "an interaction needs at least two backgrounds; build the cells "
            "with build_interventions(..., across=...)"
        )

    frame = table.head(top)
    labels = [f"{lever.split('.')[-1]} = {value}"
              for lever, value in zip(frame["lever"], frame["value"])]
    y = np.arange(len(frame))[::-1]
    colours = series_colors(len(background_cols))

    with styled():
        fig, ax = plt.subplots(figsize=(FIG_DOUBLE_COL, 0.42 * len(frame) + 1.7))
        ax.axvline(0, color=CHROME["ink"], lw=0.8, zorder=0)

        values = np.column_stack([frame[c].to_numpy() for c in background_cols])
        for row, yi in zip(values, y):
            finite = row[np.isfinite(row)]
            if len(finite) > 1:
                ax.plot([finite.min(), finite.max()], [yi, yi],
                        color=CHROME["muted"], lw=1.0, zorder=1)
        # shrinking marker sizes so a lever with *no* interaction — markers at
        # the same x — reads as concentric rings rather than as one series
        # silently hidden under another, which looks like missing data
        for k, (name, colour) in enumerate(zip(background_cols, colours)):
            ax.scatter(values[:, k], y, s=52 - 16 * k, c=colour, edgecolors="white",
                       linewidths=0.6, zorder=2 + k,
                       label=name.split("=", 1)[-1] if "=" in name else name)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"effect on {outcome_label}" if outcome_label else "lever effect")
        ax.set_title(title)
        legend_title = (background_cols[0].split("=")[0].split(".")[-1]
                        if "=" in background_cols[0] else "background")
        ax.legend(frameon=False, fontsize="small", title=legend_title,
                  title_fontsize="small", loc="best")
        ax.margins(x=0.18)
    return fig
