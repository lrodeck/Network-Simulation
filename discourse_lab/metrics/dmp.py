"""Threads as digital micro publics: the DMP measurement layer (C13 Part 2).

The unit of analysis is the bounded arena formed around a single act of
publication and the responsive exchanges it triggers. The model generates
thousands per run; this module measures them. It produces a PER-THREAD
table — one row per DMP — and the distribution across that table is the
object of study, not a population mean. That is the conceptual break with
the rest of `metrics/`: every other metric is population-scoped.

Post-hoc only, per the C13 dev notes §2.1: threads are fully
reconstructible from `posts.parquet` joined to `engagements.parquet`,
nothing new is persisted, nothing runs inside the tick.

What a DMP is (§2.2): a root post with at least `min_replies` responsive
posts. A publication that triggered nothing is not a public — the initial
act AND the responsive exchanges it triggers are the object. The threshold
is theoretically loaded: it sets the sample size by an order of magnitude
(Goel: >99% of cascades terminate in one generation), so `min_replies` is a
parameter and its sensitivity should be reported, not buried.

Measures the model CANNOT produce (§2.5), stated as absent rather than
proxied: moderation (no role, no action), anonymity and identity fluidity
(authors are stable ids), cross-media linkage (no hyperlinks or citations —
permanently out of scope).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from discourse_lab.measures import gini
from discourse_lab.metrics.stylized import stance_clusters

__all__ = ["dmp_table", "dmp_thread_count"]


def _tail_exponent(sizes: np.ndarray) -> float:
    """CCDF log-log slope over subtree sizes, nan below 5 points."""
    sizes = np.asarray(sizes, dtype=float)
    sizes = sizes[sizes > 0]
    if len(sizes) < 5:
        return float("nan")
    x = np.sort(sizes)
    ranks = np.arange(1, len(x) + 1)
    ccdf = ranks / len(x)
    return float(-np.polyfit(np.log(x), np.log(ccdf), 1)[0])


def dmp_table(handle, pop, lex=None, min_replies: int = 2,
              incivility_threshold: float = 0.7) -> pl.DataFrame:
    """One row per digital micro public: a root post with >= `min_replies`
    responsive posts. Columns group by the draft's four dimensions; see the
    change notes §2.3 for the reading of each.

    `min_replies` is analysis-time on purpose (it changes no dynamics, so it
    must not fork the run cache) — pass it, don't configure it.

    Camp-dependent columns (`cross_camp_contact`, `friction_persistence`)
    are NaN, not zero, when the population is unimodal: camp is then noise,
    and the same gate the narrator and the affect metrics apply governs
    here too.
    """
    posts = handle.posts()
    if len(posts) == 0:
        return pl.DataFrame()

    id_arr = posts["id"].to_numpy()
    author = posts["author"].to_numpy()
    t_arr = posts["t"].to_numpy()
    parent = posts["parent"].to_numpy()
    root_arr = np.where(posts["root"].to_numpy() >= 0, posts["root"].to_numpy(), id_arr)
    depth = posts["depth"].to_numpy()
    kind = posts["kind"].to_numpy()
    engagement = posts["engagement_count"].to_numpy().astype(float)
    stance_cols = [c for c in posts.columns if c.startswith("stance_")]
    stance = np.column_stack([posts[c].to_numpy() for c in stance_cols]) if stance_cols else np.zeros((len(posts), 0))
    valence = posts["valence"].to_numpy()
    arousal = posts["arousal"].to_numpy()
    specificity = posts["specificity"].to_numpy()
    provoc = posts["provocativeness"].to_numpy()
    topic = posts["topic"].to_numpy()

    # children lists
    has_children = np.zeros(len(posts), dtype=bool)
    parent_nonneg = parent[parent >= 0]
    if len(parent_nonneg) > 0:
        uniq_parents, child_counts = np.unique(parent_nonneg, return_counts=True)
        has_children[np.searchsorted(id_arr, uniq_parents)] = child_counts > 0

    # camp labels under the shared bimodality gate
    camps = None
    if len(stance_cols) > 0:
        from discourse_lab.metrics.polarization import camps_and_bimodality

        pop_stance_cols = [i for i, nm in enumerate(pop.trait_names) if nm.startswith("stance_")]
        camps, _ = camps_and_bimodality(pop.X_used[:, pop_stance_cols])

    rows = []
    roots = np.unique(root_arr)
    for root_id in roots:
        idx = np.flatnonzero(root_arr == root_id)
        responsive = idx[(id_arr[idx] != root_id)]
        if len(responsive) < min_replies:
            continue
        n = len(idx)
        root_i = int(np.flatnonzero(id_arr == root_id)[0])

        # -- dimension 1: actors and practices -----------------------------
        participants = np.unique(author[idx])
        n_participants = len(participants)
        per_author = np.bincount(np.searchsorted(participants, author[idx]))
        initiator_share = float((author[idx] == author[root_i]).mean())
        participation_gini = gini(per_author)
        return_rate = float((per_author > 1).mean())

        # -- dimension 2: meaning -------------------------------------------
        # near-trivial today (replies inherit topic) — informative only if
        # reply topic drift is ever allowed (C13 dev notes §6, out of scope)
        p_topic = topic[idx]
        span = np.linspace(-1, 1, max(len(idx), 1))
        topic_coherence = float(1.0 - np.std(p_topic) / max(np.std(span), 1e-9)) if len(idx) > 2 else float("nan")
        stance_range = float((stance[idx].max(axis=0) - stance[idx].min(axis=0)).mean()) if stance.shape[1] else float("nan")
        specificity_mean = float(specificity[idx].mean())

        # -- dimension 3: resonance ------------------------------------------
        # affective trajectory: slope of valence/arousal over depth (the
        # by-depth and by-tick curves the draft asks for collapse to their
        # linear read for the per-thread table)
        d_idx = depth[idx]
        valence_slope = _slope(d_idx, valence[idx])
        arousal_slope = _slope(d_idx, arousal[idx])
        incivility_rate = float((provoc[idx] > incivility_threshold).mean())

        # cross-camp contact + friction persistence (nan without camps)
        cross_contact = float("nan")
        friction = float("nan")
        if camps is not None:
            child_of = {int(id_arr[i]): i for i in idx}
            cross_pairs, same_pairs = [], []
            for i in idx:
                p = parent[i]
                if p < 0:
                    continue
                pi = child_of.get(int(p))
                if pi is None:
                    continue
                is_cross = camps[author[i]] != camps[author[pi]]
                # does the exchange continue past this pair: does the CHILD
                # have a child of its own?
                (cross_pairs if is_cross else same_pairs).append(bool(has_children[i]))
            if cross_pairs and same_pairs:
                friction = float(np.mean(cross_pairs) / max(np.mean(same_pairs), 1e-9))
            elif cross_pairs:
                friction = float("inf")
            cross_contact = float(np.mean(cross_pairs)) if (cross_pairs or same_pairs) else float("nan")

        # -- dimension 4: structure and power ---------------------------------
        # agenda_setting (§2.3: "the one to build first"): coefficient of
        # root stance in a regression of reply stance on root stance, per
        # depth, pooled across stance axes. The DECAY over depth is the
        # interesting shape — a DMP where root stance still predicts stance
        # at depth 3 is a different object from one where influence dies at
        # depth 1.
        agenda = {f"agenda_d{d}": float("nan") for d in (1, 2, 3)}
        for d in (1, 2, 3):
            sel = idx[(depth[idx] == d) & (id_arr[idx] != root_id)]
            if len(sel) >= 3 and stance.shape[1] > 0:
                yr = stance[sel].ravel()
                xr = np.tile(stance[root_i], len(sel))
                xr_c = xr - xr.mean()
                yr_c = yr - yr.mean()
                denom = float(xr_c @ xr_c)
                agenda[f"agenda_d{d}"] = float((xr_c @ yr_c) / denom) if denom > 1e-9 else float("nan")

        own_eng = engagement[idx]
        visibility_gini = gini(own_eng)
        terminal_fraction = float((~has_children[idx]).mean())

        # subtree sizes within the thread: descendants per post
        sub_sizes = _subtree_sizes(parent, id_arr, idx)
        subtree_alpha = _tail_exponent(sub_sizes)
        lifespan = int(t_arr[idx].max() - t_arr[root_i])

        rows.append({
            "root": int(root_id), "root_author": int(author[root_i]), "root_t": int(t_arr[root_i]),
            "n_posts": int(n), "n_responsive": int(len(responsive)),
            # dimension 1
            "n_participants": n_participants,
            "initiator_share": initiator_share,
            "participation_gini": participation_gini,
            "return_rate": return_rate,
            # dimension 2
            "topic_coherence": topic_coherence,
            "stance_range": stance_range,
            "specificity_mean": specificity_mean,
            # dimension 3
            "valence_depth_slope": valence_slope,
            "arousal_depth_slope": arousal_slope,
            "incivility_rate": incivility_rate,
            "cross_camp_contact": cross_contact,
            "friction_persistence": friction,
            # dimension 4
            **agenda,
            "visibility_gini": visibility_gini,
            "terminal_fraction": terminal_fraction,
            "subtree_alpha": subtree_alpha,
            "lifespan": lifespan,
        })

    return pl.DataFrame(rows)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-9:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _subtree_sizes(parent: np.ndarray, id_arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Descendant count per post within the thread (sub-tree sizes)."""
    id_set = {int(i): k for k, i in zip(idx, id_arr[idx])}
    children: dict[int, list[int]] = {}
    for i in idx:
        p = parent[i]
        if p >= 0 and int(p) in id_set:
            children.setdefault(int(p), []).append(i)
    sizes = []
    for i in idx:
        stack = list(children.get(int(id_arr[i]), []))
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(children.get(int(id_arr[node]), []))
        sizes.append(count)
    return np.asarray(sizes, dtype=float)


def dmp_thread_count(handle, min_replies: int = 2) -> int:
    """How many DMPs a run produced at the given threshold — the sample-size
    sensitivity check §2.2 asks for."""
    posts = handle.posts()
    if len(posts) == 0:
        return 0
    root_arr = np.where(posts["root"].to_numpy() >= 0, posts["root"].to_numpy(),
                        posts["id"].to_numpy())
    _, counts = np.unique(root_arr, return_counts=True)
    return int((counts - 1 >= min_replies).sum())
