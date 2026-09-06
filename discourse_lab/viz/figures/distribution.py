"""Tail figures: engagement, cascade size, Lorenz curves."""

from __future__ import annotations

from discourse_lab.viz._deps import require_matplotlib
from discourse_lab.viz.style import CHROME, FIG_SINGLE_COL, series_colors, styled


def fig_engagement_ccdf(engagement, fit=None, title="Engagement per post"):
    """Log-log CCDF with the fitted exponent annotated — and, when the tail
    is not a power law, said so on the figure.

    A straight line on these axes is the claim spec §5.1 makes. Printing an
    alpha beside a curve that visibly bends is how a non-result gets read as
    a result, so the annotation carries `alpha_spread` and the verdict.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    from discourse_lab.metrics.powerlaw import ccdf, powerlaw_fit

    engagement = np.asarray(engagement, dtype=float)
    fit = fit if fit is not None else powerlaw_fit(engagement)
    values, survival = ccdf(engagement[engagement >= 1])
    data_colour, fit_colour = series_colors(2)

    with styled():
        fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL, 2.6))
        ax.loglog(values, survival, color=data_colour, label="observed")

        if fit and np.isfinite(fit.xmin):
            tail = values >= fit.xmin
            if tail.any():
                x = values[tail]
                y = survival[tail][0] * (x / x[0]) ** (-(fit.alpha - 1))
                ax.loglog(x, y, color=fit_colour, linewidth=1.5,
                          label=f"fit  $\\alpha$={fit.alpha:.2f}")
            verdict = ("power law" if fit.is_powerlaw
                       else f"NOT a power law\n($\\alpha$ spread {fit.alpha_spread:.2f} over $x_{{min}}$)")
            ax.text(0.03, 0.06, verdict, transform=ax.transAxes, fontsize=6.5,
                    color=CHROME["secondary"], va="bottom")

        ax.set_xlabel("engagements per post")
        ax.set_ylabel("P(X $\\geq$ x)")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right")
    return fig


def fig_cascade_sizes(root, title="Cascade size"):
    """Log-log CCDF of cascade sizes with the singleton share annotated —
    the share is the §5.1 fact, and it lives at x=1 where a log axis makes it
    least readable, so it is stated in words."""
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    from discourse_lab.metrics.powerlaw import ccdf
    from discourse_lab.metrics.stylized import cascade_singleton_share, cascade_sizes

    sizes = cascade_sizes(np.asarray(root))
    share = cascade_singleton_share(np.asarray(root))
    values, survival = ccdf(sizes.astype(float))
    (colour,) = series_colors(1)

    with styled():
        fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL, 2.6))
        ax.loglog(values, survival, color=colour)
        ax.set_xlabel("posts in cascade")
        ax.set_ylabel("P(X $\\geq$ x)")
        ax.set_title(title, loc="left")
        ax.text(0.97, 0.95, f"singletons: {share:.1%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=CHROME["ink"])
    return fig


def fig_lorenz(series: dict, title="Concentration"):
    """Lorenz curves with the equality diagonal in chrome and each Gini
    direct-labelled. `series` maps a name to a value array."""
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    from discourse_lab.measures import gini
    from discourse_lab.metrics.stylized import lorenz_curve

    colours = series_colors(len(series))
    with styled():
        fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL, 2.8))
        ax.plot([0, 1], [0, 1], color=CHROME["baseline"], linestyle="--", linewidth=1.0)
        for (name, values), colour in zip(series.items(), colours):
            values = np.asarray(values, dtype=float)
            population, share = lorenz_curve(values)
            ax.plot(population, share, color=colour, label=f"{name}  G={gini(values):.2f}")
        ax.set_xlabel("cumulative share of population")
        ax.set_ylabel("cumulative share of total")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left")
    return fig
