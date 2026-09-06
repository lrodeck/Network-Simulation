"""Graph structure against a degree-matched null (spec §5.1)."""

from __future__ import annotations

from discourse_lab.viz._deps import require_matplotlib
from discourse_lab.viz.style import CHROME, FIG_SINGLE_COL, series_colors, styled


def fig_degree_ccdf(graph, rng=None, title="In-degree"):
    """In-degree CCDF against a degree-preserving rewiring.

    The null here is `configuration_null`, which preserves both degree
    sequences exactly by double-edge swaps — so on this figure the two curves
    coincide by construction. That is the point: it shows the degree
    distribution is not what any clustering or homophily result rests on.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    from discourse_lab.metrics.powerlaw import ccdf, powerlaw_fit
    from discourse_lab.network.measures import configuration_null

    rng = rng if rng is not None else np.random.default_rng(0)
    observed = np.asarray(graph.sum(axis=0)).ravel().astype(float)
    null = np.asarray(configuration_null(graph, rng).sum(axis=0)).ravel().astype(float)
    obs_colour, null_colour = series_colors(2)
    fit = powerlaw_fit(observed)

    with styled():
        fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL, 2.6))
        for values, colour, label in ((null, null_colour, "degree-matched null"),
                                      (observed, obs_colour, "observed")):
            v, s = ccdf(values[values >= 1])
            ax.loglog(v, s, color=colour, label=label)
        ax.set_xlabel("in-degree (followers)")
        ax.set_ylabel("P(X $\\geq$ x)")
        ax.set_title(title, loc="left")
        if fit:
            ax.text(0.03, 0.06, f"$\\alpha$={fit.alpha:.2f}", transform=ax.transAxes,
                    fontsize=7, color=CHROME["secondary"])
        ax.legend(loc="upper right")
    return fig
