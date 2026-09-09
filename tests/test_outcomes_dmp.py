"""Experiment 02 P2 verification: cascade reconstruction and per-cascade
DMP outcomes, on hand-built data so the numbers can be checked by hand.
"""

from __future__ import annotations

import polars as pl
import pytest

from discourse_lab.outcomes_dmp import (
    audience_members,
    cascade_windows,
    dmp_outcomes,
    enacted_members,
)


def _posts():
    # Two cascades: root 0 (users 1,2,3 post; branches to depth 2), root 10
    # (a singleton -- author 4 only, no replies).
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 10],
            "t": [5, 6, 8, 20],
            "author": [1, 2, 3, 4],
            "parent": [-1, 0, 1, -1],
            "root": [0, 0, 0, 10],
            "depth": [0, 1, 2, 0],
        }
    )


def _exposures():
    return pl.DataFrame(
        {
            "t": [5, 6, 7, 20],
            "user": [2, 5, 6, 7],
            "post": [0, 0, 1, 10],
            "attended": [True, False, True, True],
        }
    )


def _traits():
    # snapshots at t=0,5,10,15,20 -- dense enough to bracket root 0's window
    # [5,8] with before_t=5, after_t=10 (distinct).
    rows = []
    for t in (0, 5, 10, 15, 20):
        for user in range(1, 8):
            rows.append({"t": t, "user": user, "stance_0": 0.1 * user + 0.01 * t, "animus": 0.2 + 0.001 * t})
    return pl.DataFrame(rows)


def test_cascade_windows_matches_hand_count():
    windows = cascade_windows(_posts())
    w = windows.sort("root")
    assert w["root"].to_list() == [0, 10]
    assert w["size"].to_list() == [3, 1]
    assert w["t_start"].to_list() == [5, 20]
    assert w["t_end"].to_list() == [8, 20]
    assert w["max_depth"].to_list() == [2, 0]


def test_enacted_members_are_the_posting_authors():
    m = enacted_members(_posts()).sort(["root", "user"])
    assert m.filter(pl.col("root") == 0)["user"].sort().to_list() == [1, 2, 3]
    assert m.filter(pl.col("root") == 10)["user"].sort().to_list() == [4]


def test_audience_members_join_exposures_through_post_to_root():
    m = audience_members(_exposures(), _posts()).sort(["root", "user"])
    # post 0 and post 1 both belong to root 0 -> users 2,5,6 exposed to root 0
    assert m.filter(pl.col("root") == 0)["user"].sort().to_list() == [2, 5, 6]
    assert m.filter(pl.col("root") == 10)["user"].sort().to_list() == [7]


def test_dmp_outcomes_excludes_singletons_and_matches_hand_computed_delta():
    out = dmp_outcomes(_posts(), _exposures(), _traits(), membership="enacted")
    # root 10 is a singleton (size 1) -- excluded per MIN_CASCADE_SIZE.
    assert out["root"].to_list() == [0]
    row = out.row(0, named=True)
    assert row["size"] == 3
    assert row["before_t"] == 5
    assert row["after_t"] == 10
    # enacted members of root 0: users 1,2,3. animus(t)=0.2+0.001t exactly,
    # so every user's delta_animus_i = 0.001*(10-5) = 0.005 regardless of user.
    assert row["delta_animus"] == pytest.approx(0.005, abs=1e-9)
    # stance_0(t) = 0.1*user + 0.01*t -> delta_i = 0.01*(10-5) = 0.05 for every user
    assert row["delta_stance"] == pytest.approx(0.05, abs=1e-9)
    assert row["n_members"] == 3


def test_dmp_outcomes_audience_differs_from_enacted_membership():
    enacted = dmp_outcomes(_posts(), _exposures(), _traits(), membership="enacted")
    audience = dmp_outcomes(_posts(), _exposures(), _traits(), membership="audience")
    assert enacted.row(0, named=True)["n_members"] == 3  # users 1,2,3
    assert audience.row(0, named=True)["n_members"] == 3  # users 2,5,6 -- different set, same count here


def test_dmp_outcomes_rejects_too_coarse_a_snapshot_schedule():
    posts = _posts()
    exposures = _exposures()
    # Only two global snapshots, like the Experiment 01 FULL runs (t=0,
    # t=T-1). Root 0's own lifespan is [5,8] (width 3), but the only
    # available bracket is [0,20] (width 20) -- "distinct" (0 != 20) but
    # 6.7x wider than the cascade it's supposed to measure, well past the
    # default 5x threshold. Must refuse rather than silently return a
    # global-window delta mislabeled as this cascade's own.
    coarse_traits = pl.DataFrame(
        {"t": [0, 20] * 7, "user": sorted(list(range(1, 8)) * 2),
         "stance_0": [0.0] * 14, "animus": [0.2] * 14}
    )
    with pytest.raises(ValueError, match="too coarse"):
        dmp_outcomes(posts, exposures, coarse_traits, membership="enacted")


def test_dmp_outcomes_rejects_unknown_membership():
    with pytest.raises(ValueError, match="membership must be"):
        dmp_outcomes(_posts(), _exposures(), _traits(), membership="bogus")
