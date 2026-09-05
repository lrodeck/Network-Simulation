"""Single workspace resolver — nothing hardcoded (dev notes §2.2)."""

from __future__ import annotations

import os
from pathlib import Path


def workspace() -> Path:
    p = os.environ.get("DLAB_HOME")
    if p:
        return Path(p)
    if Path("/content").exists():          # Colab
        return Path("/content/dlab")
    return Path.cwd() / "dlab"


def runs_dir() -> Path:
    return workspace() / "runs"


def artifacts_dir() -> Path:
    return workspace() / "artifacts"


def scenarios_dir() -> Path:
    return workspace() / "scenarios"


def voice_cards_dir() -> Path:
    return workspace() / "voice_cards"


def ensure_workspace() -> Path:
    for d in (runs_dir(), artifacts_dir(), scenarios_dir(), voice_cards_dir()):
        d.mkdir(parents=True, exist_ok=True)
    return workspace()
