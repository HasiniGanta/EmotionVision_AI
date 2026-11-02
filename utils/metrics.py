"""utils/metrics.py - lightweight timing/statistics helpers used across the app."""

import time
from contextlib import contextmanager


@contextmanager
def timer():
    """Usage: with timer() as t: ...  then t['elapsed_ms']"""
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0


def format_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


def safe_divide(a, b, default=0.0):
    try:
        return a / b if b else default
    except Exception:
        return default
