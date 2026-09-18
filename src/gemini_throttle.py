"""Serialize Gemini calls so Free-tier RPM limits are not burst."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, TypeVar

from src.config import GEMINI_MIN_INTERVAL_SEC, GEMINI_RETRY_WAIT_SEC

T = TypeVar("T")

_lock = threading.Lock()
_last_call = 0.0


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "resource exhausted" in text
        or "too many requests" in text
        or ("rate" in text and "limit" in text)
    )


def gemini_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a Gemini API function with spacing + 429 retries."""
    last_exc: BaseException | None = None
    for attempt in range(3):
        with _lock:
            global _last_call
            now = time.monotonic()
            wait = GEMINI_MIN_INTERVAL_SEC - (now - _last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                result = fn(*args, **kwargs)
                _last_call = time.monotonic()
                return result
            except Exception as exc:
                _last_call = time.monotonic()
                last_exc = exc
                retry = attempt < 2 and _is_rate_limit_error(exc)
        if retry:
            time.sleep(GEMINI_RETRY_WAIT_SEC)
            continue
        assert last_exc is not None
        raise last_exc
    assert last_exc is not None
    raise last_exc
