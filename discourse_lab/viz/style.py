"""The figure design system (dev notes; palette validated, see below).

One system for every figure, so a reader can carry an encoding from one plot
to the next. The rules that are not negotiable:

- Categorical hues are assigned in fixed slot order and never cycled. A ninth
  series is not a generated hue — `series_colors` raises and points at
  faceting.
- Sequential is one hue, light to dark. Diverging is two hues with a *neutral
  gray* midpoint, built symmetric about zero so gray lands exactly on it.
- No dual axes anywhere. Two measures of different scale get two panels.
- Dashing is reserved for threshold lines (R_eff = 1, a target band edge), so
  a dashed line always means "this is not data".
- Status colors are reserved for pass/fail and always ship beside a word,
  never as color alone.

The palette passes all six checks of the validator at surface #fcfcfb:
lightness band, chroma floor, adjacent-pair CVD separation (worst 9.1 protan),
normal-vision floor (worst 19.6), with a contrast WARN on three slots that is
discharged by the direct labels and legends every figure carries and by the
tables in `viz.tables`.
"""

from __future__ import annotations

from contextlib import contextmanager

# fixed slot order — never cycled, never reordered by a filter
PALETTE = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)

CHROME = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
}

# reserved: pass/fail only, always with a word beside them
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}

SEQUENTIAL_HUE = ("#e8f0fb", "#2a78d6", "#12365f")   # light -> dark, one hue
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")        # blue <-> red, neutral gray middle

FIG_SINGLE_COL = 3.5    # inches
FIG_DOUBLE_COL = 7.0


def series_colors(n: int) -> list[str]:
    """The first `n` slots, in order. Raises above 8 rather than inventing a
    hue: a ninth series is a signal to facet or fold into "other"."""
    if n > len(PALETTE):
        raise ValueError(
            f"{n} series exceeds the {len(PALETTE)}-slot categorical palette. "
            "Facet into small multiples or fold the tail into 'other' — "
            "generating a ninth hue makes the pair indistinguishable."
        )
    return list(PALETTE[:n])


def diverging_cmap():
    """Blue-gray-red, for values symmetric about zero. Pair with
    `symmetric_norm` so the neutral gray sits exactly on zero."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("dlab_diverging", DIVERGING)


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("dlab_sequential", SEQUENTIAL_HUE)


def symmetric_norm(values):
    """`vmin = -vmax` so a diverging ramp's midpoint is zero, not the data
    mean. Without this the gray drifts and the sign of a cell stops being
    readable from its color."""
    import numpy as np
    from matplotlib.colors import Normalize

    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    limit = float(v.max()) if len(v) and v.max() > 0 else 1.0
    return Normalize(vmin=-limit, vmax=limit)


RC = {
    "figure.facecolor": CHROME["surface"],
    "axes.facecolor": CHROME["surface"],
    "savefig.facecolor": CHROME["surface"],
    "axes.edgecolor": CHROME["muted"],
    "axes.labelcolor": CHROME["ink"],
    "axes.titlecolor": CHROME["ink"],
    "text.color": CHROME["ink"],
    "xtick.color": CHROME["secondary"],
    "ytick.color": CHROME["secondary"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": CHROME["grid"],
    "grid.linestyle": "-",          # solid hairline; dashes mean thresholds
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 4.0,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,             # journals require embedded TrueType
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
}


@contextmanager
def styled():
    """Scoped rcParams. Never mutates the global state — a notebook that
    imports this must not find its own plots restyled."""
    import matplotlib.pyplot as plt

    with plt.rc_context(RC):
        yield
