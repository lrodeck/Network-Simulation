"""The world a run was conducted in — as important as the results.

A config hash and a JSON dump say a run is reproducible; they do not say what
world it describes. These render the setup: the ideological axes and their
shapes, the archetype mixture, and — the one that matters most and is hardest
to see — how the traits actually relate to each other.
"""

from __future__ import annotations

from discourse_lab.viz._deps import require_matplotlib
from discourse_lab.viz.style import (
    CHROME,
    FIG_DOUBLE_COL,
    FIG_SINGLE_COL,
    diverging_cmap,
    series_colors,
    styled,
    symmetric_norm,
)

# spec §1.1's column blocks, in table order
BLOCKS = ("personality", "expression", "topic_affinity", "stance", "behavior", "meta")


def _block_of(name: str) -> str:
    if name.startswith("topic_affinity_"):
        return "topic_affinity"
    if name.startswith("stance_"):
        return "stance"
    if name in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        return "personality"
    if name in ("verbosity", "formality", "irony", "humor", "profanity", "emoji"):
        return "expression"
    if name in ("activity", "reply_prop", "repost_prop", "contrarianism", "credulity", "prominence"):
        return "behavior"
    return "meta"


def fig_scenario_axes(cfg, title=None):
    """The ideological space: each stance axis's empirical marginal, with its
    poles named on the axis.

    This is what a scenario *is* — the JSON carries a 128-bin density per axis
    and a pair of pole labels, and reading it as JSON tells you nothing about
    whether the population is bimodal, skewed, or centrist. The shape is the
    modelling claim.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    axes_spec = cfg.scenario.stance_axes
    if not axes_spec:
        raise ValueError(
            "no scenario attached, so there are no named axes to draw. "
            "Use discourse_lab.data.scenario_config() to attach one."
        )
    colours = series_colors(min(len(axes_spec), 8))

    with styled():
        fig, panels = plt.subplots(
            1, len(axes_spec), figsize=(FIG_DOUBLE_COL, 2.0), squeeze=False
        )
        for ax, spec, colour in zip(panels[0], axes_spec, colours):
            marginal = spec["marginal"]
            lo, hi = marginal["support"]
            density = np.asarray(marginal["density"], dtype=float)
            grid = np.linspace(lo, hi, len(density))

            ax.fill_between(grid, density, color=colour, alpha=0.25, linewidth=0)
            ax.plot(grid, density, color=colour)
            ax.axvline(0.0, color=CHROME["baseline"], linestyle="--", linewidth=1.0)

            ax.set_title(spec.get("name", "?"), loc="left", fontsize=8)
            ax.set_xticks([lo, 0, hi])
            ax.set_xticklabels([spec.get("pole_neg", "-"), "", spec.get("pole_pos", "+")],
                               fontsize=7)
            ax.set_yticks([])
            ax.set_ylabel("density" if ax is panels[0][0] else "")
        fig.suptitle(title or f"Ideological axes — scenario '{cfg.scenario.name}'",
                     x=0.01, ha="left", fontsize=9)
    return fig


def fig_trait_correlations(pop, title="How traits actually relate"):
    """The realised correlation between traits in the sampled population.

    **Realised, not requested, and they are not the same.** `correlation_pairs`
    defaults to empty, which yields an identity matrix — so a reader of the
    config would conclude the traits are independent. Measured on the shipped
    default at N=8000, they are not: `activity x reply_prop = +0.30` and
    `plasticity x conviction = -0.11`, induced entirely by the archetype
    mixture, because an archetype that shifts two traits together correlates
    them. "The loudest users are also the most argumentative" is a substantive
    modelling claim that appears nowhere in the config.

    Diverging ramp with a neutral grey midpoint on a symmetric norm, so zero is
    grey and the sign of a cell is readable from its colour.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    names = list(pop.trait_names)
    corr = np.corrcoef(pop.X_stored.T)
    off = corr - np.eye(len(names))

    with styled():
        fig, ax = plt.subplots(figsize=(FIG_DOUBLE_COL, FIG_DOUBLE_COL * 0.82))
        image = ax.imshow(off, cmap=diverging_cmap(), norm=symmetric_norm(off))

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=5.5)
        ax.set_yticklabels(names, fontsize=5.5)
        ax.grid(False)

        # block separators: without them a 31x31 grid of small labels is
        # unreadable, and the blocks are what spec §1.1 actually defines
        blocks = [_block_of(n) for n in names]
        edges = [i for i in range(1, len(blocks)) if blocks[i] != blocks[i - 1]]
        for e in edges:
            ax.axhline(e - 0.5, color=CHROME["muted"], linewidth=0.6)
            ax.axvline(e - 0.5, color=CHROME["muted"], linewidth=0.6)

        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        bar.set_label("Pearson r (diagonal removed)", fontsize=7)
        bar.ax.tick_params(labelsize=6)

        strongest = np.unravel_index(np.argmax(np.abs(off)), off.shape)
        ax.set_title(
            f"{title}  —  strongest: {names[strongest[0]]} × {names[strongest[1]]} "
            f"= {off[strongest]:+.2f}",
            loc="left", fontsize=8,
        )
    return fig


def fig_archetypes(cfg, pop, title="Archetype mixture"):
    """Who is in this world: requested weights against realised membership,
    and what each archetype shifts.

    Membership is a label the dynamics never read (spec §2.1) — it only shapes
    the initial draw — so this is a picture of the *inputs*, and the offsets
    panel is where the induced trait correlations above come from.
    """
    require_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    from discourse_lab.population.archetypes import resolve_archetypes

    archetypes = resolve_archetypes(
        cfg.population.archetype_weights, cfg.population.archetype_offsets
    )
    names = [a.name for a in archetypes]
    requested = np.array([a.w for a in archetypes], dtype=float)
    counts = np.bincount(pop.archetype_labels, minlength=len(names)).astype(float)
    realised = counts / counts.sum()

    shifted = sorted({trait for a in archetypes for trait in a.offsets})
    grid = np.zeros((len(names), len(shifted)))
    for i, a in enumerate(archetypes):
        for trait, value in a.offsets.items():
            grid[i, shifted.index(trait)] = value

    req_colour, real_colour = series_colors(2)
    with styled():
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(FIG_DOUBLE_COL, 2.4), gridspec_kw={"width_ratios": [1, 1.3]}
        )
        y = np.arange(len(names))
        left.barh(y - 0.19, requested, height=0.36, color=req_colour, label="requested")
        left.barh(y + 0.19, realised, height=0.36, color=real_colour, label="realised")
        left.set_yticks(y)
        left.set_yticklabels(names, fontsize=7)
        left.set_xlabel("share of population")
        # upper right: the mixture is dominated by lurkers, so the long bar is
        # always at the bottom and a legend there sits on top of it
        left.legend(loc="upper right")
        left.set_title("mixture weights", loc="left", fontsize=8)

        image = right.imshow(grid, cmap=diverging_cmap(), norm=symmetric_norm(grid), aspect="auto")
        right.set_xticks(range(len(shifted)))
        right.set_xticklabels(shifted, rotation=45, ha="right", fontsize=6.5)
        right.set_yticks(y)
        right.set_yticklabels(names, fontsize=7)
        right.grid(False)
        right.set_title("trait offsets (latent units)", loc="left", fontsize=8)
        bar = fig.colorbar(image, ax=right, fraction=0.046, pad=0.03)
        bar.ax.tick_params(labelsize=6)

        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    return fig
