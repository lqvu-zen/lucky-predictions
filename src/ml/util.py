"""Small shared helpers for the ML modules."""
from __future__ import annotations

import sys


def progress(i: int, n: int, label: str = "") -> None:
    """Draw a one-line progress bar on stderr — but only when stderr is an
    interactive terminal. In logs / CI (not a TTY) it prints nothing, so
    output stays clean.
    """
    try:
        if not sys.stderr.isatty():
            return
    except Exception:
        return
    n = max(n, 1)
    width = 22
    filled = int(width * i / n)
    bar = "#" * filled + "." * (width - filled)
    sys.stderr.write(f"\r  {label} [{bar}] {i}/{n}")
    sys.stderr.flush()
    if i >= n:
        sys.stderr.write("\n")
        sys.stderr.flush()
