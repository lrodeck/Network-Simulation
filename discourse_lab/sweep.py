"""Parallel sweeps over (config, seed) pairs (dev notes §7.4, §8.1).

The cache is the coordination mechanism: each worker checks `cached_run`
before executing, so an interrupted sweep resumes by re-invocation and
duplicate work is skipped without any locking.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from discourse_lab.config import Config
from discourse_lab.runner import cached_run


def free_threaded() -> bool:
    return hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()


def interpreter_pool_available() -> bool:
    return sys.version_info >= (3, 14) and hasattr(
        __import__("concurrent.futures", fromlist=["InterpreterPoolExecutor"]),
        "InterpreterPoolExecutor",
    )


def fork_or_spawn_usable() -> bool:
    try:
        import multiprocessing

        multiprocessing.get_start_method()
        return True
    except (ImportError, ValueError):
        return False


class SequentialExecutor(Executor):
    """In-process fallback: no pool, no pickling, worst tracebacks avoided."""

    def submit(self, fn, /, *args, **kwargs) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - propagate via Future
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        pass


def executor(n_workers: int) -> Executor:
    if os.environ.get("DLAB_WORKERS") == "1":
        return SequentialExecutor()
    if n_workers <= 1:
        return SequentialExecutor()
    if free_threaded():
        from concurrent.futures import ThreadPoolExecutor

        return ThreadPoolExecutor(n_workers)
    if interpreter_pool_available():
        from concurrent.futures import InterpreterPoolExecutor

        return InterpreterPoolExecutor(n_workers)
    if fork_or_spawn_usable():
        return ProcessPoolExecutor(n_workers)
    return SequentialExecutor()


def work_queue(configs: Sequence[Config], seeds: Sequence[int]) -> Iterator[tuple[Config, int]]:
    """Flat queue, seed-major (dev §8.1): seed 1 of every config, then seed 2, ..."""
    for s in seeds:
        for cfg in configs:
            yield cfg, s


def _run_one(cfg: Config, seed: int) -> Path:
    return cached_run(cfg, seed)


def sweep(
    configs: Sequence[Config],
    seeds: Sequence[int],
    n_workers: int = 1,
) -> Iterable[Path]:
    """Run every (config, seed) pair, seed-major, skipping cached cells.

    Workers return paths, not Run objects (dev §7.4): pickling a
    multi-gigabyte Run back through a pipe costs more than recomputing it.
    """
    pairs = list(work_queue(configs, seeds))
    with executor(n_workers) as ex:
        futures = [ex.submit(_run_one, cfg, seed) for cfg, seed in pairs]
        for fut in futures:
            yield fut.result()
