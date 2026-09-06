"""Writing figures out at publication quality.

`constrained_layout`, never `bbox_inches="tight"`: tight-bbox changes the
saved figure's dimensions, so `\\includegraphics[width=\\columnwidth]` then
scales it by an amount that varies per figure and the font sizes stop matching
across a paper.
"""

from __future__ import annotations

from pathlib import Path

from discourse_lab.io.workspace import figures_dir


def save_figure(fig, name: str, directory: Path | None = None, formats=("pdf", "png")) -> dict[str, Path]:
    """Write `name.pdf` and `name.png`; returns the paths by format.

    PDF is the one that goes in the paper (vector, TrueType-embedded via
    `pdf.fonttype = 42`); PNG is for notebooks and quick looks.
    """
    directory = Path(directory) if directory is not None else figures_dir()
    directory.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    for fmt in formats:
        path = directory / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        out[fmt] = path
    return out
