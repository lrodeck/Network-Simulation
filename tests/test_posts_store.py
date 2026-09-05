"""Persistence of the raw per-post and per-engagement tables.

Before this, a run directory held only 9 per-tick scalars, so four of the
five spec §5.1 facts with encoded ranges had no data source at all.
"""

from __future__ import annotations

import dataclasses
import json
import warnings

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.io.store import RUN_FORMAT, RunHandle
from discourse_lab.runner import cached_run, load_run, run, run_dir, run_iter


def _cfg(n_users=300, n_ticks=8):
    return dataclasses.replace(
        Config(),
        population=dataclasses.replace(Config().population, n_users=n_users),
        dynamics=dataclasses.replace(Config().dynamics, n_ticks=n_ticks, drift="none"),
    )


@pytest.fixture(autouse=True)
def _quiet_cascade_warnings():
    # tiny demo populations make the cascade caps bind constantly; that is
    # expected at this scale and not what these tests are about
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="cascade:")
        yield


def test_every_post_reaches_disk_exactly_once(tmp_path, monkeypatch):
    """Posts are written as they retire, so the end-of-run flush is the easy
    thing to get wrong — anything still active on the last tick would be
    silently dropped.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = _cfg()

    issued = set()
    for state in run_iter(cfg, seed=0):
        if state.retired_posts is not None:
            issued.update(int(i) for i in state.retired_posts.id)

    run(cfg, seed=0, persist=("posts",))
    posts = load_run(cfg, seed=0).posts()

    assert posts.height == len(issued)
    assert set(posts["id"].to_list()) == issued
    # ids are handed out contiguously, so a gap means a lost post
    assert issued == set(range(max(issued) + 1))


def test_posts_schema_and_stance_flattening(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = _cfg()
    run(cfg, seed=0, persist=("posts",))
    posts = load_run(cfg, seed=0).posts()

    for column in ("id", "t", "author", "topic", "parent", "root", "depth", "kind", "engagement_count"):
        assert column in posts.columns
    for dim in ("arousal", "valence", "provocativeness", "novelty", "specificity", "quality", "length"):
        assert dim in posts.columns
    # stance is 2-D in memory and must be flattened per axis to fit parquet
    for d in range(cfg.stance_dims()):
        assert f"stance_{d}" in posts.columns

    assert set(posts["kind"].unique()) <= {"post", "repost", "quote", "reply"}
    assert (posts["engagement_count"] >= 0).all()


def test_engagements_are_the_spec_event_log(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = _cfg()
    run(cfg, seed=0, persist=("posts", "engagements"))
    handle = load_run(cfg, seed=0)

    engagements = handle.engagements()
    assert engagements.columns == ["t", "user", "post", "action"]  # spec §1.5
    assert "skip" not in set(engagements["action"].unique())  # skip is the reference category

    # every engagement points at a post that was actually written
    assert set(engagements["post"].to_list()) <= set(handle.posts()["id"].to_list())

    # and the per-post totals reconcile with the counts carried on the posts
    counted = engagements.group_by("post").len().sort("post")
    posts = handle.posts().sort("id")
    lookup = dict(zip(posts["id"].to_list(), posts["engagement_count"].to_list()))
    for post_id, n in zip(counted["post"].to_list(), counted["len"].to_list()):
        assert lookup[post_id] == n


def test_persistence_is_opt_in_and_absence_is_reported_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = _cfg(n_ticks=4)
    run(cfg, seed=0)  # no persist=

    handle = load_run(cfg, seed=0)
    assert handle.has_posts is False
    assert handle.has_engagements is False
    with pytest.raises(FileNotFoundError, match="persist"):
        handle.posts()


def test_unknown_persist_target_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="unknown persist target"):
        run(_cfg(n_ticks=2), seed=0, persist=("psots",))


def test_cached_run_reruns_when_asked_for_posts_it_does_not_have(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = _cfg(n_ticks=4)

    cached_run(cfg, seed=0)  # first run, metrics only
    assert load_run(cfg, seed=0).has_posts is False

    cached_run(cfg, seed=0, persist=("posts",))  # now posts are wanted
    assert load_run(cfg, seed=0).has_posts is True


def test_stale_format_directory_is_rerun(tmp_path, monkeypatch):
    """An old run predating posts.parquet must not be served from cache."""
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg = _cfg(n_ticks=4)
    path = cached_run(cfg, seed=0)

    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["format"] = 1
    meta_path.write_text(json.dumps(meta))
    assert RunHandle(path).format == 1

    cached_run(cfg, seed=0)
    assert RunHandle(path).format == RUN_FORMAT


def test_run_iter_exposes_the_raw_record_without_a_writer(tmp_path, monkeypatch):
    """Interactive callers should be able to capture posts straight off the
    generator — persistence is one consumer of this, not the only route.
    """
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    states = list(run_iter(_cfg(n_ticks=6), seed=0))

    assert any(s.retired_posts is not None for s in states)
    assert any(s.engagement_events is not None for s in states)

    events = next(s.engagement_events for s in states if s.engagement_events)
    assert set(events) == {"t", "user", "post", "action"}
    assert len({len(v) for v in events.values()}) == 1  # all columns same length
