"""A deterministic narrator: what is the discourse doing, in words.

A 500-tick run is six numeric sparklines. This turns each tick into a few
sentences so a misconfigured run is obvious at tick 20 rather than at the end,
and so a completed run can be read rather than only plotted.

Deterministic by construction — spec §0.1: "Experiments on dynamics never call
an API." No LLM touches this. It is also cheap: the per-tick work is a handful
of vectorised aggregates over quantities the tick already computed, and the
string is only formatted when someone asks for it (`describe_state`), so a run
with narration on does not pay for text it never reads.

Deliberately NOT here: per-user narratives ("agent 471 is radicalising"). They
cannot be vectorised, they invite over-reading a single trajectory, and that is
exactly where someone would later bolt an LLM call into the tick loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from discourse_lab.semantics.lexicon import Lexicon

# Sarle's bimodality coefficient for a uniform distribution. Above it, the
# population plausibly has two modes; below, talking about "camps" is noise.
BIMODAL_THRESHOLD = 5.0 / 9.0


@dataclass(frozen=True)
class DiscourseSummary:
    """Numbers only. Formatting happens in `describe_state`, so the tick loop
    never builds a string it might not use."""

    t: int
    top_topics: tuple[int, ...]
    top_shares: tuple[float, ...]
    dominant_axis: tuple[int, ...]        # per top topic, the max-|sigma| axis
    dominant_sign: tuple[float, ...]
    dominant_mag: tuple[float, ...]
    hardening: tuple[float, ...]          # EWMA change in |sigma| per topic
    bimodality: float
    has_camps: bool
    camp_sizes: tuple[int, int]
    camp_separation: float
    camp_separation_delta: float
    cross_cutting_share: float
    cross_cutting_hostility: float
    attention_gini: float
    r_eff: float


@dataclass
class Narrator:
    """Pure observer. Holds only the history needed for trend arrows, so it can
    be attached to `run_iter` or replayed over a finished run.

    The EWMA lives here rather than on `TickEngine` deliberately: the engine is
    the model, and a trend line is a property of watching it, not of it.
    """

    lex: Lexicon
    ewma: float = 0.3
    _prev_mag: np.ndarray | None = field(default=None, init=False)
    _prev_sep: float | None = field(default=None, init=False)
    _hardening: np.ndarray | None = field(default=None, init=False)

    def observe(
        self,
        t: int,
        s: np.ndarray,
        sigma: np.ndarray,
        stance_u: np.ndarray,
        users: np.ndarray | None = None,
        authors: np.ndarray | None = None,
        actions: np.ndarray | None = None,
        metrics: dict | None = None,
        n_top: int = 2,
    ) -> DiscourseSummary:
        from discourse_lab.metrics import bimodality_coefficient, cluster_centroid_distance
        from discourse_lab.metrics.stylized import inter_cluster_interaction, stance_clusters

        metrics = metrics or {}
        s = np.asarray(s, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        share = s / s.sum() if s.sum() > 0 else np.zeros_like(s)
        order = np.argsort(share)[::-1][:n_top]

        mag = np.abs(sigma)                       # (K, D)
        axis_of = mag.argmax(axis=1) if mag.size else np.zeros(len(s), dtype=int)
        per_topic_mag = mag.max(axis=1) if mag.size else np.zeros(len(s))

        if self._hardening is None:
            self._hardening = np.zeros_like(per_topic_mag)
        if self._prev_mag is not None:
            delta = per_topic_mag - self._prev_mag
            self._hardening = self.ewma * delta + (1 - self.ewma) * self._hardening
        self._prev_mag = per_topic_mag.copy()

        # Camps are only meaningful if the population is actually bimodal.
        # `stance_clusters` splits on the first principal component and so
        # returns two groups even for a unimodal cloud, where the split — and
        # every "polarization" number derived from it — is noise.
        projection = stance_u @ _first_component(stance_u) if stance_u.shape[1] > 1 else stance_u[:, 0]
        bimodality = float(bimodality_coefficient(projection))
        has_camps = bool(bimodality > BIMODAL_THRESHOLD)

        labels = stance_clusters(stance_u)
        sizes = (int((labels == 0).sum()), int((labels == 1).sum()))
        separation = float(cluster_centroid_distance(stance_u, labels))
        sep_delta = separation - self._prev_sep if self._prev_sep is not None else 0.0
        self._prev_sep = separation

        if users is not None and authors is not None and actions is not None and len(users):
            rate, hostility = inter_cluster_interaction(users, authors, actions, labels)
        else:
            rate, hostility = float("nan"), float("nan")

        return DiscourseSummary(
            t=int(t),
            top_topics=tuple(int(k) for k in order),
            top_shares=tuple(float(share[k]) for k in order),
            dominant_axis=tuple(int(axis_of[k]) for k in order),
            dominant_sign=tuple(float(np.sign(sigma[k, axis_of[k]])) for k in order),
            dominant_mag=tuple(float(mag[k, axis_of[k]]) for k in order),
            hardening=tuple(float(self._hardening[k]) for k in order),
            bimodality=bimodality,
            has_camps=has_camps,
            camp_sizes=sizes,
            camp_separation=separation,
            camp_separation_delta=float(sep_delta),
            cross_cutting_share=float(rate),
            cross_cutting_hostility=float(hostility),
            attention_gini=float(metrics.get("attention_gini", float("nan"))),
            r_eff=float(metrics.get("r_eff", float("nan"))),
        )


def _first_component(x: np.ndarray) -> np.ndarray:
    centred = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0]


def _trend(value: float, eps: float = 1e-3) -> str:
    if not np.isfinite(value) or abs(value) < eps:
        return "steady"
    return "hardening" if value > 0 else "softening"


def describe_state(summary: DiscourseSummary, lex: Lexicon) -> str:
    """Four to six sentences. One line per idea, joined — readable in a log,
    in a widget, or in a notebook cell."""
    parts = []

    if summary.top_topics:
        topics = ", ".join(
            f"{lex.topic_label(k)} ({share:.0%})"
            for k, share in zip(summary.top_topics, summary.top_shares)
        )
        parts.append(f"t={summary.t}: attention on {topics}.")

    # only the strongest axis per topic — reporting all D turns this into a
    # wall of text at D=3 and worse beyond
    for k, d, sign, mag, hard in zip(
        summary.top_topics, summary.dominant_axis, summary.dominant_sign,
        summary.dominant_mag, summary.hardening,
    ):
        if mag < 1e-6:
            continue
        parts.append(
            f"On {lex.topic_label(k)} the {lex.pole_label(d, sign)} pole of "
            f"{lex.axis_names[d]} leads (|σ|={mag:.2f}, {_trend(hard)})."
        )

    if summary.has_camps:
        drift = ""
        if abs(summary.camp_separation_delta) > 1e-3:
            direction = "up" if summary.camp_separation_delta > 0 else "down"
            drift = f", {direction} {abs(summary.camp_separation_delta):.2f}"
        # No camp sizes here: `stance_clusters` splits at the median of the
        # dominant component, so the two groups are exactly N/2 by
        # construction. Printing "400/400" as though it were a measurement
        # invites reading a structural artifact as a finding.
        parts.append(
            f"Population is bimodal ({summary.bimodality:.2f}); the two halves sit "
            f"{summary.camp_separation:.2f} apart{drift}."
        )
    else:
        parts.append(
            f"No clear camps (bimodality {summary.bimodality:.2f} ≤ {BIMODAL_THRESHOLD:.2f}); "
            "cross-camp numbers below are a median split, not a cleavage."
        )

    if np.isfinite(summary.cross_cutting_share):
        hostility = ""
        if np.isfinite(summary.cross_cutting_hostility):
            hostility = f", {summary.cross_cutting_hostility:.0%} of them hostile"
        parts.append(
            f"{summary.cross_cutting_share:.0%} of engagements cross camps{hostility}."
        )

    flags = []
    if np.isfinite(summary.r_eff) and summary.r_eff > 1:
        flags.append(f"R_eff={summary.r_eff:.2f} > 1 (cascades supercritical)")
    if np.isfinite(summary.attention_gini) and summary.attention_gini > 0.9:
        flags.append(f"attention Gini {summary.attention_gini:.2f}")
    if flags:
        parts.append("Flags: " + "; ".join(flags) + ".")

    return " ".join(parts)


def describe_run(handle, lex: Lexicon | None = None, at: Sequence[int] | None = None) -> str:
    """Narrate a finished run from its persisted tables.

    Recomputed rather than read back from a stored narrative: persisting the
    text would freeze today's wording into every cached run, and a narration
    improvement could never reach runs already on disk.
    """
    lex = lex if lex is not None else Lexicon.from_handle(handle)
    metrics = handle.metrics()
    ticks = metrics["t"].to_list()
    at = list(at) if at is not None else ticks[:: max(1, len(ticks) // 5)]

    lines = []
    for t in at:
        row = metrics.filter(metrics["t"] == t)
        if len(row) == 0:
            continue
        lines.append(
            f"t={t}: attention Gini {row['attention_gini'][0]:.3f}, "
            f"R_eff {row['r_eff'][0]:.3f}, "
            f"{int(row['n_posts'][0])} new posts, "
            f"{int(row['n_engagements'][0])} engagements."
        )
    return "\n".join(lines)
