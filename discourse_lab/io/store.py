"""Streaming parquet persistence and the run artifact (dev notes §5, §7.3).

Layout of one run:

    dlab/runs/{cfg_hash}/{seed}/
        meta.json          full config, serialised
        metrics.parquet    one row group per tick
        traits.parquet     per-tick snapshots, added when population lands
        ...
        COMPLETE           marker written on clean shutdown

Snapshots stream to parquet incrementally and are never accumulated in memory,
so memory per worker stays flat in run length. Exposures will not be persisted
in full — they outnumber engagements roughly 50:1 (spec §3.5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


class RunWriter:
    def __init__(self, path: Path, cfg, seed: int):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.seed = seed
        self._writers: dict[str, pq.ParquetWriter] = {}
        meta = {
            "config": cfg.to_dict(),
            "config_json": cfg.to_json(),
            "config_hash": cfg.hash(),
            "seed": int(seed),
            "sub_hashes": cfg.sub_hashes(),
            "format": 1,
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

    def write_tick(self, t: int, metrics_row: dict[str, float], snapshot: bool = False) -> None:
        row = {"t": t, **{k: float(v) for k, v in metrics_row.items()}}
        self._writer("metrics", self._batch(row).schema).write_batch(self._batch(row))
        # traits snapshots hook here once the population phase lands (dev §7.3):
        # self._writer("traits", ...).write_batch(snapshot_batch)

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

    @property
    def complete(self) -> bool:
        return (self.path / "COMPLETE").exists()

    def metrics(self) -> pl.DataFrame:
        return pl.read_parquet(self.path / "metrics.parquet")

    def ticks(self) -> np.ndarray:
        return self.metrics()["t"].to_numpy()
