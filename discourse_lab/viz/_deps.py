"""matplotlib is an optional dependency (`pip install -e '.[viz]'`).

The core package must never import it: a sweep on a headless box has no reason
to carry a plotting stack, and `discourse_lab.runner` importing pyplot would
make that unavoidable. Every figure module calls `require_matplotlib()` before
importing pyplot so the failure is a sentence rather than a traceback.
"""

from __future__ import annotations


def require_matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by hand
        raise ModuleNotFoundError(
            "matplotlib is required for discourse_lab.viz figures. "
            "Install it with:  pip install -e '.[viz]'"
        ) from exc
    return matplotlib
