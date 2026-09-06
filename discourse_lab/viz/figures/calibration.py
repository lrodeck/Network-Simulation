"""The calibration figure (spec §5.1): every fact against its target band."""

from __future__ import annotations

from discourse_lab.viz._deps import require_matplotlib
from discourse_lab.viz.style import CHROME, FIG_DOUBLE_COL, STATUS, styled


def fig_calibration(report: dict, title: str = "Stylized facts vs spec §5.1 targets"):
    """Horizontal dot-per-fact against its target band.

    Facts are on wildly different scales (a Gini in [0,1], a clustering ratio
    of 4.5, an exponent near 6), so each is plotted as its position *within
    its own band* rather than on a shared axis — a shared axis here would be
    the dual-axis mistake in another costume. The raw value is printed beside
    every dot, which is also what discharges the palette's contrast warning.

    Status is never colour alone: each row carries the word pass/FAIL/n/a.
    Ungraded rows (a target the spec does not quote, or an exponent whose
    tail is not a power law) are drawn in chrome, not in a status colour.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    items = list(report.items())
    labels = [entry["label"] for _, entry in items]
    n = len(items)

    with styled():
        fig, ax = plt.subplots(figsize=(FIG_DOUBLE_COL, 0.42 * n + 1.0))
        for i, (_, entry) in enumerate(items):
            y = n - 1 - i
            target = entry.get("target")
            value = float(entry["value"])
            in_range = entry.get("in_range")

            if in_range is None:
                colour, word = CHROME["secondary"], "n/a"
            elif in_range:
                colour, word = STATUS["good"], "pass"
            else:
                colour, word = STATUS["critical"], "FAIL"

            # the value always prints, graded or not
            ax.text(1.42, y, f"{value:.3g}  {word}", va="center", ha="left",
                    color=CHROME["ink"], fontsize=7.5)

            if target is None:
                # No band to place a dot in. Drawing one anyway puts a mark at
                # an arbitrary position that a reader will inevitably read as
                # a score — the printed value is the whole content of the row.
                ax.text(0.5, y, "no target quoted", va="center", ha="center",
                        fontsize=6.5, color=CHROME["muted"], style="italic")
                continue

            lo, hi = (target[0], target[0] * 2) if not np.isfinite(target[1]) else target
            span = hi - lo if hi > lo else 1.0
            ax.barh(y, 1.0, left=0.0, height=0.55, color=CHROME["grid"], zorder=1)
            position = float(np.clip((value - lo) / span, -0.35, 1.35))

            ax.plot([position], [y], "o", color=colour, markersize=7, zorder=3,
                    markeredgecolor=CHROME["surface"], markeredgewidth=1.0)

        ax.set_yticks(range(n))
        ax.set_yticklabels(labels[::-1], fontsize=7.5)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["target min", "target max"], fontsize=7)
        ax.set_xlim(-0.45, 2.1)
        ax.set_ylim(-0.7, n - 0.3)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(CHROME["baseline"])
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", pad=8)
    return fig
