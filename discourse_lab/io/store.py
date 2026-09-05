"""Streaming parquet persistence and the run artifact (dev notes §5, §7.3).

Layout of one run:

    dlab/runs/{cfg_hash}/{seed}/
        meta.json            full config, serialised
        metrics.parquet      one row group per tick
        posts.parquet        one row per post, written as posts retire
        engagements.parquet  the (user, post, action, t) event log (spec §1.5)
        COMPLETE             marker written on clean shutdown

Snapshots stream to parquet incrementally and are never accumulated in memory,
so memory per worker stays flat in run length. Exposures are not persisted in
full — they outnumber engagements roughly 50:1 (spec §3.5).

`posts` and `engagements` are optional: pass `persist=` to `run()`. They are
deliberately *not* config fields — persistence is an output concern, and
putting it in `DynamicsConfig` would change `cfg.hash()` and invalidate every
cached artifact and run directory in existence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Bumped when the on-disk layout changes. `cached_run` only reuses a directory
# written at this version or later, so runs predating posts.parquet are
# recomputed once (population and graph artifacts are keyed on their own
# sub-hashes and survive, so only the dynamics re-run).
RUN_FORMAT = 2

POST_DIM_COLUMNS = (
    "arousal", "valence", "provocativeness", "novelty", "specificity", "quality", "length",
)
ENGAGEMENT_COLUMNS = ("t", "user", "post", "action")


def posts_schema(stance_dims: int) -> pa.Schema:
    """Pre-declared so every row group matches — `ParquetWriter` requires an
    identical schema per batch, and stance has to be flattened to columns
    because `pa.array` rejects a 2-D array.
    """
    fields = [
        ("id", pa.int64()), ("t", pa.int64()), ("author", pa.int64()), ("topic", pa.int64()),
        ("parent", pa.int64()), ("root", pa.int64()), ("depth", pa.int64()),
        ("kind", pa.string()), ("engagement_count", pa.int64()),
    ]
    fields += [(name, pa.float64()) for name in POST_DIM_COLUMNS]
    fields += [(f"stance_{d}", pa.float64()) for d in range(stance_dims)]
    return pa.schema(fields)


def engagements_schema() -> pa.Schema:
    return pa.schema([("t", pa.int64()), ("user", pa.int64()), ("post", pa.int64()), ("action", pa.string())])


def posts_batch(posts, stance_dims: int) -> pa.RecordBatch:
    cols: dict[str, Any] = {
        "id": posts.id, "t": posts.t, "author": posts.author, "topic": posts.topic,
        "parent": posts.parent, "root": posts.root, "depth": posts.depth,
        "kind": [str(k) for k in posts.kind], "engagement_count": posts.engagement_count,
    }
    for name in POST_DIM_COLUMNS:
        cols[name] = np.asarray(getattr(posts, name), dtype=np.float64)
    for d in range(stance_dims):
        cols[f"stance_{d}"] = np.asarray(posts.stance[:, d], dtype=np.float64)

    schema = posts_schema(stance_dims)
    arrays = [pa.array(cols[field.name], type=field.type) for field in schema]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


class RunWriter:
    def __init__(self, path: Path, cfg, seed: int):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.seed = seed
        self.stance_dims = cfg.stance_dims()
        self._writers: dict[str, pq.ParquetWriter] = {}
        meta = {
            "config": cfg.to_dict(),
            "config_json": cfg.to_json(),
            "config_hash": cfg.hash(),
            "seed": int(seed),
            "sub_hashes": cfg.sub_hashes(),
            "format": RUN_FORMAT,
        }
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _writer(self, name: str, schema: pa.Schema) -> pq.ParquetWriter:
        w = self._writers.get(name)
        if w is None:
            w = pq.ParquetWriter(self.path / f"{name}.parquet", schema)
            self._writers[name] = w
        return w

    @staticmethod
    def _batch(table: dict[str, Any]) -> pa.RecordBatch:
        cols = {}
        for k, v in table.items():
            v = np.asarray(v)
            if v.ndim == 0:
                v = v.reshape(1)
            cols[k] = v
        names = list(cols)
        arrays = [pa.array(cols[k]) for k in names]
        schema = pa.schema([(n, arrays[i].type) for i, n in enumerate(names)])
        return pa.RecordBatch.from_arrays(arrays, schema=schema)

    def write_table(self, name: str, schema: pa.Schema, batch: pa.RecordBatch) -> None:
        """Generic streaming append — one row group per call. The traits
        snapshot hook (dev §7.3) lands on this too when it arrives.
        """
        self._writer(name, schema).write_batch(batch)

    def write_tick(self, t: int, metrics_row: dict[str, float]) -> None:
        row = {"t": t, **{k: float(v) for k, v in metrics_row.items()}}
        self._writer("metrics", self._batch(row).schema).write_batch(self._batch(row))

    def write_posts(self, posts) -> None:
        if posts is None or len(posts) == 0:
            return
        self.write_table("posts", posts_schema(self.stance_dims), posts_batch(posts, self.stance_dims))

    def write_engagements(self, events: dict[str, np.ndarray] | None) -> None:
        if not events or len(events["user"]) == 0:
            return
        schema = engagements_schema()
        arrays = [
            pa.array(np.asarray(events["t"]), type=pa.int64()),
            pa.array(np.asarray(events["user"]), type=pa.int64()),
            pa.array(np.asarray(events["post"]), type=pa.int64()),
            pa.array([str(a) for a in events["action"]], type=pa.string()),
        ]
        self.write_table("engagements", schema, pa.RecordBatch.from_arrays(arrays, schema=schema))

    def ensure_empty(self, name: str, schema: pa.Schema) -> None:
        """Create the file even if nothing was ever written, so `has_posts`
        distinguishes "persistence was off" from "nothing happened".
        """
        self._writer(name, schema)

    def close(self) -> None:
        for w in self._writers.values():
            w.close()
        self._writers.clear()
        (self.path / "COMPLETE").write_text("ok", encoding="utf-8")


class RunHandle:
    """Lazy view over a persisted run; analysis reads what it needs."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        self.meta: dict[str, Any] = meta
        self.config_json: str = meta["config_json"]
        self.config_hash: str = meta["config_hash"]
        self.seed: int = meta["seed"]
        self.format: int = int(meta.get("format", 1))

    @property
    def complete(self) -> bool:
        return (self.path / "COMPLETE").exists()

    def metrics(self) -> pl.DataFrame:
        return pl.read_parquet(self.path / "metrics.parquet")

    def ticks(self) -> np.ndarray:
        return self.metrics()["t"].to_numpy()

    @property
    def has_posts(self) -> bool:
        return (self.path / "posts.parquet").exists()

    @property
    def has_engagements(self) -> bool:
        return (self.path / "engagements.parquet").exists()

    def _require(self, name: str, flag: str) -> Path:
        p = self.path / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} does not exist — this run was written without {name} persistence. "
                f'Re-run with run(cfg, seed, persist=("{flag}",)) to produce it.'
            )
        return p

    def posts(self) -> pl.DataFrame:
        return pl.read_parquet(self._require("posts", "posts"))

    def engagements(self) -> pl.DataFrame:
        return pl.read_parquet(self._require("engagements", "engagements"))
