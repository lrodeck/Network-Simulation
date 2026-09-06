"""Experiment 1 effects against the matched null (spec §5.2, §5.3)."""

from __future__ import annotations

from discourse_lab.viz._deps import require_matplotlib
from discourse_lab.viz.style import CHROME, FIG_DOUBLE_COL, series_colors, styled


def fig_effect_dots(rows, metrics=("agreement_effect", "salience_effect", "bubble_index_effect"),
                    title="Kernel effect vs matched null"):
    """Dot-and-interval per (kernel, ranker), one panel per metric.

    The zero line is the whole point — spec §5.3 exists because an effect
    that does not clear its null is not an effect — so it is drawn in chrome
    and every interval is +/- 1 SD across seeds. Points, not bars: these are
    signed differences, and a bar implies a magnitude from a meaningful zero
    baseline which is exactly what is being tested rather than assumed.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl

    frame = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(list(rows))
    metrics = [m for m in metrics if m in frame.columns]
    keys = [c for c in ("kernel", "ranker") if c in frame.columns]

    agg = (
        frame.group_by(keys)
        .agg([pl.col(m).mean().alias(m) for m in metrics]
             + [pl.col(m).std().alias(f"{m}__sd") for m in metrics])
        .sort(keys)
    )
    labels = [" / ".join(str(r[k]) for k in keys) for r in agg.iter_rows(named=True)]
    colours = series_colors(min(len(metrics), 8))

    with styled():
        fig, axes = plt.subplots(1, len(metrics), figsize=(FIG_DOUBLE_COL, 0.34 * len(labels) + 1.4),
                                 sharey=True, squeeze=False)
        y = np.arange(len(labels))
        for ax, metric, colour in zip(axes[0], metrics, colours):
            mean = agg[metric].to_numpy().astype(float)
            sd = np.nan_to_num(agg[f"{metric}__sd"].to_numpy().astype(float))
            ax.axvline(0.0, color=CHROME["baseline"], linewidth=1.0)
            ax.errorbar(mean, y, xerr=sd, fmt="o", color=colour, markersize=5,
                        elinewidth=1.5, capsize=0,
                        markeredgecolor=CHROME["surface"], markeredgewidth=0.8)
            ax.set_title(metric.replace("_effect", "").replace("_", " "), loc="left", fontsize=8)
        axes[0][0].set_yticks(y)
        axes[0][0].set_yticklabels(labels, fontsize=7)
        axes[0][0].set_ylim(-0.6, len(labels) - 0.4)
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    return fig
