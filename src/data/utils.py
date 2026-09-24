"""Lightweight cached HTTP helper for public football APIs."""

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

try:
    from zoneinfo import ZoneInfo

    _SAST = ZoneInfo("Africa/Johannesburg")
    _UTC = timezone.utc
except Exception:
    _SAST = timezone(timedelta(hours=2))
    _UTC = timezone.utc

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_TIMEOUT = 20

# Cache lifetimes by data volatility — finished seasons never change,
# live/same-day data changes by the minute.
TTL_LIVE = 300                # in-play / same-day data
TTL_CURRENT = 1800            # current-season fixtures & standings
TTL_HISTORICAL = 7 * 86400    # completed seasons & static archives

# One session per worker thread so keep-alive is reused without sharing a
# Session across threads.
_thread_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s


def _cache_key(url: str, ext: str = ".json") -> str:
    return hashlib.sha256(url.encode()).hexdigest() + ext


# Serialises concurrent fetches of the same URL so parallel callers share
# one HTTP request instead of racing each other (and the cache file).
_CACHE_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


def _cache_lock(key: str) -> threading.Lock:
    with _CACHE_LOCKS_GUARD:
        return _CACHE_LOCKS.setdefault(key, threading.Lock())


def _is_cache_valid(cache_path: Path, ttl_seconds: int) -> bool:
    if not cache_path.exists():
        return False
    try:
        age = time.time() - cache_path.stat().st_mtime
        return age < ttl_seconds
    except Exception:
        return False


def _write_cache(cache_path: Path, data: Any, raw: bool) -> None:
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if raw:
            f.write(data)
        else:
            json.dump(data, f)
    os.replace(tmp, cache_path)


def cached_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    ttl_seconds: int = 3600,
    timeout: int = DEFAULT_TIMEOUT,
    raw: bool = False,
) -> Any:
    """
    Perform a GET request with a simple on-disk cache.

    Args:
        url: URL to fetch.
        headers: Optional request headers.
        ttl_seconds: Cache time-to-live.
        timeout: Request timeout.
        raw: If True, return the response text instead of parsed JSON.

    Returns:
        Parsed JSON dict, response text, or None if the request fails.
    """
    ext = ".txt" if raw else ".json"
    key = _cache_key(url, ext=ext)
    cache_path = CACHE_DIR / key

    with _cache_lock(key):
        if _is_cache_valid(cache_path, ttl_seconds):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return f.read() if raw else json.load(f)
            except Exception:
                pass

        try:
            response = _session().get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.text if raw else response.json()
            _write_cache(cache_path, data, raw)
            return data
        except Exception as exc:
            logger.warning(f"Failed to fetch {url}: {exc}")
            # Serve stale cache if it exists
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        return f.read() if raw else json.load(f)
                except Exception:
                    pass
    return None


def clear_cache() -> int:
    """Remove all cached JSON files. Returns number removed."""
    removed = 0
    for f in CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def cache_age(url: str) -> Optional[float]:
    """Return the age in seconds of a cached URL, or None."""
    key = _cache_key(url)
    cache_path = CACHE_DIR / key
    if not cache_path.exists():
        return None
    return time.time() - cache_path.stat().st_mtime


def to_sast(dt: Optional[datetime], source_tz_name: str = "UTC") -> Optional[datetime]:
    """
    Convert a match datetime to South African Standard Time (SAST, UTC+2).

    Args:
        dt: Source datetime, either naive or aware.
        source_tz_name: IANA timezone name the naive value is in, or "UTC".

    Returns:
        A naive SAST datetime (no tzinfo), or None if dt is None.
    """
    if dt is None:
        return None
    try:
        if dt.tzinfo is None:
            if source_tz_name == "UTC":
                dt = dt.replace(tzinfo=_UTC)
            else:
                dt = dt.replace(tzinfo=ZoneInfo(source_tz_name))
        sast = dt.astimezone(_SAST).replace(tzinfo=None)
        return sast
    except Exception:
        # If IANA data is unavailable, add a fixed +2 hour offset as a fallback.
        return dt.replace(tzinfo=None) + timedelta(hours=2) if dt.tzinfo is None else dt.astimezone(_SAST).replace(tzinfo=None)


def sast_now() -> datetime:
    """Return the current time in SAST as a naive datetime."""
    try:
        return datetime.now(_SAST).replace(tzinfo=None)
    except Exception:
        return datetime.now(timezone(timedelta(hours=2))).replace(tzinfo=None)
