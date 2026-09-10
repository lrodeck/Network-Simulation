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


def emergent_camps(stance: np.ndarray, k_max: int = 5, seed: int = 0) -> dict:
    """V6(2) (change spec V1-V6): k as a MEASUREMENT, not an assumption.

    `camps_and_bimodality` median-splits ONE axis, so k=2 is hardcoded before
    camps are defined at all. This fits k-means at k=1..k_max and selects k
    by BIC (Pelleg & Moore 2000's formulation: one POOLED isotropic variance
    shared across all k clusters, not one variance per cluster), so an
    intervention that dissolves two camps into five smaller hostile ones is
    visible as a change in k — the binary frame can only report that as
    "bimodality fell" and read it as success.

    Per-cluster variance was tried first and rejected: a Gaussian mixture's
    likelihood is unbounded as any one cluster's variance shrinks to 0, so
    letting BIC over-segment a true cluster into a tiny low-variance sliver
    is rewarded rather than penalized, and the score decreased monotonically
    to k_max instead of turning back up past the true k. Pooling the
    variance across clusters removes that degenerate degree of freedom.

    The log-likelihood also needs the mixing-proportion term
    `sum_i n_i * log(n_i / n)` (each point's probability includes which
    cluster it fell into, not just its distance from that cluster's
    centroid) — omitting it was the second failure mode tried: BIC still
    picked k_max, because nothing was penalizing a split into many small,
    uneven clusters when only the distance term was scored. With it,
    `n_params = k*d + 1 + (k-1)` (centroids, one shared variance, k-1 free
    mixing proportions) — the standard k-means BIC (Pelleg & Moore 2000).

    Full stance geometry (all D axes), never a single dominant projection —
    the mechanism-level requirement V3 also needs (a user close on axis 1
    and far on axis 0 should not be flattened onto one line before k is
    even chosen). Returns `{"k", "labels", "bic", "centroids"}`; `bic` is
    the winning k's score, for comparability across runs/interventions.
    """
    from scipy.cluster.vq import kmeans2

    n, d = stance.shape
    best: dict | None = None
    for k in range(1, min(k_max, n) + 1):
        if k == 1:
            centroids = stance.mean(axis=0, keepdims=True)
            labels = np.zeros(n, dtype=int)
        else:
            centroids, labels = kmeans2(stance, k, seed=seed, minit="++")

        rss = float(((stance - centroids[labels]) ** 2).sum())
        free_dims = max(n - k, 1)
        var = max(rss / (free_dims * d), 1e-12)  # pooled isotropic variance, shared across clusters

        log_lik = -0.5 * n * d * np.log(2 * np.pi * var) - rss / (2 * var)
        counts = np.bincount(labels, minlength=k)
        counts = counts[counts > 0]
        log_lik += float(np.sum(counts * np.log(counts / n)))  # mixing-proportion term
        n_params = k * d + 1 + (k - 1)  # centroids + shared variance + free mixing proportions
        bic = -2 * log_lik + n_params * np.log(n)

        if best is None or bic < best["bic"]:
            best = {"k": k, "labels": labels.copy(), "bic": float(bic), "centroids": centroids.copy()}

    return best


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
