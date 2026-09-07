"""Affective polarization metrics (change spec C1.4) and the shared camp
helper.

The model represents ideological position; the dominant finding in the
political-communication literature is that *affective* polarization
(inter-party animus) rose while ideological positions moved comparatively
little (Iyengar, Sood & Lelkes 2012; Iyengar & Westwood 2015). Törnberg
(PNAS 2022) locates the mechanism in partisan sorting producing animus, not
opinion divergence. These metrics give the model the outcome variable that
literature measures.

Camp definition is shared, not duplicated: the change spec is explicit that
camp is the sign on the dominant axis of stance variation (`stance_clusters`)
and that where the population is unimodal, camp is noise — affect metrics
must then be reported as **undefined (nan), not zero**, the same gate the
narrator applies (BIMODAL_THRESHOLD = 5/9, Sarle's coefficient for uniform).
"""

from __future__ import annotations

import numpy as np

from discourse_lab.metrics.stylized import _dominant_projection, stance_clusters

# Sarle's bimodality coefficient for a uniform distribution; the narrator
# uses the same gate. Above it, "two camps" describes the population; below
# it, camp membership is a projection artifact and affect comparisons are
# undefined rather than zero.
CAMP_BIMODAL_THRESHOLD = 5.0 / 9.0


def camps_and_bimodality(stance: np.ndarray) -> tuple[np.ndarray | None, float]:
    """`(labels, bimodality)` with labels = None when the population is
    unimodal — the shared gate every camp-conditional number must go through.
    """
    from discourse_lab.metrics import bimodality_coefficient  # lazy: metrics/__init__ imports this module

    projection = _dominant_projection(stance)
    bimodality = float(bimodality_coefficient(projection))
    if not np.isfinite(bimodality) or bimodality <= CAMP_BIMODAL_THRESHOLD:
        return None, bimodality
    return stance_clusters(stance), bimodality


def affective_distance(animus: np.ndarray, labels: np.ndarray | None) -> float:
    """Unnormalized between-camp `animus` gap: how much more hostile the
    average camp member is toward the other side than toward their own —
    the affective-polarization magnitude proper. Undefined (nan) when camps
    are undefined; the change spec is explicit it must not be reported as
    zero.

    Deliberately NOT normalized by the population mean: `animus_asymmetry`
    already reports the signed, mean-normalized share (that is what a
    growing-but-symmetric population needs), and dividing this magnitude by
    a population mean that itself moves under drift would report a widening
    absolute gap as flat whenever the whole population got more hostile
    together — exactly the case a symmetric kernel like `outrage` produces
    when it does not yet condition on camp.
    """
    if labels is None:
        return float("nan")
    means = [animus[labels == lbl].mean() for lbl in np.unique(labels) if (labels == lbl).any()]
    if len(means) < 2:
        return float("nan")
    return float(max(means) - min(means))


def animus_asymmetry(animus: np.ndarray, labels: np.ndarray | None) -> float:
    """Camp-wise mean animus difference (camp 1 minus camp 0), as a share of
    the population mean.

    The FIES-series papers find the two sides are rarely symmetric; a model
    that can only produce symmetric outcomes cannot speak to that literature,
    so the asymmetry is its own number rather than folded into a distance.
    Sign is arbitrary (camp labels come from a projection); magnitude is the
    claim. Undefined when camps are undefined.
    """
    if labels is None:
        return float("nan")
    a = animus[labels == 0]
    b = animus[labels == 1]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    total = float(np.mean(animus))
    if total <= 0:
        return 0.0
    return float((b.mean() - a.mean()) / total)


def ident_animus_coupling(identification: np.ndarray, animus: np.ndarray) -> float:
    """Cross-user correlation of identification with animus — the *sorting*
    signature (Törnberg 2022): camps that sort together come to feel together.
    Reported regardless of camp bimodality, since it is a population-level
    covariance, not a camp-conditional comparison.
    """
    if len(identification) < 3:
        return float("nan")
    if np.std(identification) == 0 or np.std(animus) == 0:
        return float("nan")
    return float(np.corrcoef(identification, animus)[0, 1])


def expressed_vs_latent_bimodality(
    expressed_projection: np.ndarray, latent_projection: np.ndarray
) -> dict[str, float]:
    """C7's false-consensus signature: Sarle's bimodality of *posted* stances
    against the population's latent stances.

    Under the spiral-of-silence gate, low-conviction dissenters post less, so
    the expressed distribution shifts toward the dominant camp and its
    bimodality drops below the latent value. `expressed - latent == 0` on a
    bimodal population means the gate never bound — the same "equal means the
    mechanism is decorative" reading the C2 selection test uses.
    """
    from discourse_lab.metrics import bimodality_coefficient  # lazy: avoid the package-init cycle

    latent = float(bimodality_coefficient(np.asarray(latent_projection, dtype=float)))
    expressed = float(bimodality_coefficient(np.asarray(expressed_projection, dtype=float)))
    return {
        "expressed_bimodality": expressed,
        "latent_bimodality": latent,
        "false_consensus_gap": expressed - latent,
    }
