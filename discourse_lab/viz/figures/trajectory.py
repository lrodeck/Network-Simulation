"""Per-tick metric trajectories against the matched null (spec §5.3)."""

from __future__ import annotations

from discourse_lab.viz._deps import require_matplotlib
from discourse_lab.viz.style import CHROME, FIG_DOUBLE_COL, series_colors, styled

DEFAULT_METRICS = ("attention_gini", "salience", "agreement", "bubble_index", "r_eff")


def fig_metric_trajectories(handle, null_handle, metrics=DEFAULT_METRICS, title=None):
    """Small multiples, one panel per measure, model against its null.

    `null_handle` is required, not optional: spec §5.3 makes the null
    comparison mandatory ("Heavy-tailed activity alone manufactures most of
    what naively looks like emergent structure"), and a signature that lets
    you omit it invites exactly the plot that section warns about.

    Each panel keeps its own y-axis — the measures are on different scales,
    and forcing them onto one would be the dual-axis mistake. Only two series
    per panel, so both are direct-labelled and the legend is not load-bearing.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    model, null = handle.metrics(), null_handle.metrics()
    metrics = [m for m in metrics if m in model.columns]
    colour_model, colour_null = series_colors(2)

    with styled():
        fig, axes = plt.subplots(
            1, len(metrics), figsize=(FIG_DOUBLE_COL, 2.1), sharex=True, squeeze=False
        )
        for ax, metric in zip(axes[0], metrics):
            ax.plot(null["t"], null[metric], color=colour_null, label="null")
            ax.plot(model["t"], model[metric], color=colour_model, label="model")
            ax.set_title(metric.replace("_", " "), loc="left", fontsize=8)
            ax.set_xlabel("tick")
            if metric == "r_eff":
                # dashes are reserved for thresholds, so this reads as
                # "supercritical above here" and never as another series
                ax.axhline(1.0, color=CHROME["baseline"], linestyle="--", linewidth=1.0)
                ax.text(0.97, 1.0, "R=1", transform=ax.get_yaxis_transform(),
                        va="bottom", ha="right", fontsize=6.5, color=CHROME["secondary"])
        axes[0][0].legend(loc="best")
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    return fig
