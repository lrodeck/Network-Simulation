"""Publication tables: LaTeX booktabs and Markdown from one source.

polars only — no matplotlib, so this works on a headless box, and no pandas,
which polars' own `to_latex` would pull in and which is not a dependency of
this project.

Both exporters consume the same `_rows()` rendering, so the LaTeX and the
Markdown can never disagree about a value or a rounding.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from discourse_lab.io.workspace import tables_dir

_LATEX_ESCAPES = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$"}


def _escape_latex(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in text)


def _fmt(value, precision: int) -> str:
    """`nan` renders as an em-dash, not as the string 'nan'. A table that
    prints 'nan' reads as a bug; '--' reads as 'not measured', which is what
    it means."""
    if value is None:
        return "--"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "--"
        return f"{value:.{precision}f}"
    return str(value)


def _rows(df: pl.DataFrame, precision: int) -> list[list[str]]:
    return [[_fmt(v, precision) for v in row] for row in df.iter_rows()]


def to_latex_booktabs(
    df: pl.DataFrame, caption: str = "", label: str = "", precision: int = 3
) -> str:
    cols = df.columns
    align = "l" + "r" * (len(cols) - 1)
    body = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_escape_latex(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    body += [" & ".join(_escape_latex(c) for c in row) + r" \\" for row in _rows(df, precision)]
    body += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        body.append(rf"\caption{{{_escape_latex(caption)}}}")
    if label:
        body.append(rf"\label{{{label}}}")
    body.append(r"\end{table}")
    return "\n".join(body)


def to_markdown(df: pl.DataFrame, precision: int = 3) -> str:
    cols = df.columns
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in _rows(df, precision)]
    return "\n".join(lines)


def save_table(
    df: pl.DataFrame, name: str, caption: str = "", label: str = "",
    precision: int = 3, directory: Path | None = None,
) -> dict[str, Path]:
    directory = Path(directory) if directory is not None else tables_dir()
    directory.mkdir(parents=True, exist_ok=True)
    out = {
        "tex": directory / f"{name}.tex",
        "md": directory / f"{name}.md",
    }
    # trailing newline: these are files a person opens in an editor and a
    # `\input{}` in a paper, both of which expect one
    out["tex"].write_text(to_latex_booktabs(df, caption, label, precision) + "\n", encoding="utf-8")
    out["md"].write_text(to_markdown(df, precision) + "\n", encoding="utf-8")
    return out


def stylized_facts_table(report: dict, attention_gini_source: str = "lifetime, per post") -> pl.DataFrame:
    """T1, the calibration table (spec §5.1).

    States which attention Gini it reports: `metrics.parquet:attention_gini`
    is a rolling measure over currently-active posts each tick, while this one
    is the lifetime total per post from `posts.parquet`. They measure
    different things and do not agree, so the table has to say which.

    Rows the model cannot be graded on carry "n/a" rather than a pass or fail
    — the engagement exponent is ungraded when the tail is not a power law,
    and grading it would report on where x_min landed.
    """
    rows = []
    for entry in report.values():
        target = entry.get("target")
        if target is None:
            target_text = "--"
        elif target[1] == float("inf"):
            target_text = f">= {target[0]:g}"
        else:
            target_text = f"{target[0]:g}-{target[1]:g}"
        in_range = entry.get("in_range")
        rows.append({
            "Fact": entry["label"],
            "Value": float(entry["value"]),
            "Target": target_text,
            "Status": "n/a" if in_range is None else ("pass" if in_range else "FAIL"),
        })
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        pl.when(pl.col("Fact").str.contains("Attention Gini"))
        .then(pl.lit(f"Attention Gini ({attention_gini_source})"))
        .otherwise(pl.col("Fact"))
        .alias("Fact")
    )


def world_table(cfg, pop=None) -> pl.DataFrame:
    """T0: the world a run was conducted in, as prose rather than a hash.

    `provenance_table` proves a run is reproducible; this says what it
    describes. Reports the requested setting and, where the two can differ,
    what the sampled population actually came out as — the distinction is the
    point. The clearest case is trait correlation: `correlation_pairs` defaults
    to empty, so the requested matrix is the identity and a reader of the
    config concludes the traits are independent, while the archetype mixture
    induces `activity x reply_prop = +0.30` because an archetype that shifts
    two traits together correlates them.
    """
    import numpy as np

    scenario = cfg.scenario
    rows = [
        {"Property": "scenario", "Requested": scenario.name or "(none)", "Realised": ""},
        {"Property": "stance axes", "Requested": str(cfg.stance_dims()),
         "Realised": ", ".join(scenario.axis_names()) or "(unnamed)"},
        {"Property": "topics", "Requested": str(cfg.population.n_topics),
         "Realised": ", ".join(scenario.topic_names) or "(unnamed)"},
        {"Property": "users", "Requested": str(cfg.population.n_users), "Realised": ""},
        {"Property": "graph", "Requested": cfg.graph.generator,
         "Realised": f"mean degree {cfg.graph.mean_degree:g}"},
        {"Property": "ranker", "Requested": cfg.dynamics.ranker, "Realised": ""},
        {"Property": "kernel", "Requested": cfg.dynamics.kernel, "Realised": ""},
    ]

    for axis, (neg, pos) in zip(scenario.axis_names(), scenario.poles()):
        rows.append({"Property": f"axis: {axis}", "Requested": f"{neg} -> {pos}", "Realised": ""})

    requested_pairs = cfg.population.correlation_pairs
    rows.append({
        "Property": "trait correlations",
        "Requested": f"{len(requested_pairs)} pair(s)" if requested_pairs else "none (identity)",
        "Realised": "",
    })
    if pop is not None:
        corr = np.corrcoef(pop.X_stored.T)
        off = corr - np.eye(len(pop.trait_names))
        i, j = np.unravel_index(np.argmax(np.abs(off)), off.shape)
        rows[-1]["Realised"] = (
            f"strongest {pop.trait_names[i]} x {pop.trait_names[j]} = {off[i, j]:+.2f}"
        )
    return pl.DataFrame(rows)


def experiment_effects_table(rows: Iterable[dict], metrics: Sequence[str] = ()) -> pl.DataFrame:
    """T2: kernel x ranker effects against the matched null (spec §5.3)."""
    frame = pl.DataFrame(list(rows))
    metrics = list(metrics) or [c for c in frame.columns if c.endswith("_effect")]
    keys = [c for c in ("kernel", "ranker") if c in frame.columns]
    return (
        frame.group_by(keys)
        .agg([pl.col(m).mean().alias(m) for m in metrics] +
             [pl.col(m).std().alias(f"{m}_sd") for m in metrics])
        .sort(keys)
    )


def provenance_table(meta: dict) -> pl.DataFrame:
    """T3: what produced the numbers — config sub-hashes, seed, run format."""
    rows = [
        {"Field": "config hash", "Value": meta.get("config_hash", "--")},
        {"Field": "seed", "Value": str(meta.get("seed", "--"))},
        {"Field": "run format", "Value": str(meta.get("format", "--"))},
    ]
    rows += [
        {"Field": f"sub-hash: {k}", "Value": v}
        for k, v in sorted((meta.get("sub_hashes") or {}).items())
    ]
    return pl.DataFrame(rows)
