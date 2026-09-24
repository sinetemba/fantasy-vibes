"""Lightweight cached HTTP helper for public football APIs."""

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# Expired files double as a stale-data fallback on fetch failure, so only
# prune them once they're well past usefulness.
CACHE_MAX_AGE = 30 * 86400

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
        # Bound the map — a held lock object stays valid for its holder even
        # if the entry is dropped, so clearing just loses dedup for new calls.
        if len(_CACHE_LOCKS) > 10_000:
            _CACHE_LOCKS.clear()
        return _CACHE_LOCKS.setdefault(key, threading.Lock())


# Only genuine cache artifacts are prunable — never user state like
# preferences.json or persisted model files.
_PRUNABLE = re.compile(r"^[0-9a-f]{64}\.(json|txt)$")


def prune_cache(max_age_seconds: int = CACHE_MAX_AGE) -> int:
    """Delete stale cache files (hashed URL entries, leftover .tmp files and
    the scraped PSL page) older than max_age. Returns number removed."""
    removed = 0
    cutoff = time.time() - max_age_seconds
    for f in CACHE_DIR.glob("*"):
        try:
            prunable = (
                _PRUNABLE.match(f.name)
                or f.suffix == ".tmp"
                or f.name in ("psl_matchcentre.html", "psl_log.html")
            )
            if prunable and f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            pass
    if removed:
        logger.info(f"Pruned {removed} stale cache files")
    return removed


# Live-refresh window: while open, cache entries are only trusted for
# LIVE_TTL seconds so refresh/auto-refresh pull live scores. Files are
# never deleted — other pages keep their cached data once it closes.
LIVE_TTL = 15
_LIVE_WINDOW_UNTIL = 0.0
_LIVE_WINDOW_GUARD = threading.Lock()


def bust_live_cache(seconds: float = 45) -> None:
    """Open a live-refresh window for `seconds`."""
    global _LIVE_WINDOW_UNTIL
    with _LIVE_WINDOW_GUARD:
        _LIVE_WINDOW_UNTIL = time.time() + seconds


def cache_ttl(ttl_seconds: int) -> int:
    """Effective TTL — collapsed to LIVE_TTL while a live-refresh window
    is open so live scores refetch instead of serving stale cache."""
    if time.time() < _LIVE_WINDOW_UNTIL:
        return min(ttl_seconds, LIVE_TTL)
    return ttl_seconds


prune_cache()


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
        if _is_cache_valid(cache_path, cache_ttl(ttl_seconds)):
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


def compute_table(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute a standings table from a match list. Every team appears,
    even with no recorded result; form is last-5, oldest first (the UI
    expects rightmost = latest)."""
    table: Dict[str, Dict[str, Any]] = {}
    all_teams = set()

    for m in matches:
        home = m.get("home_team")
        away = m.get("away_team")
        if not home or not away:
            continue
        all_teams.add(home)
        all_teams.add(away)

        if m.get("status") != "full_time" or m.get("home_score") is None:
            continue

        for team, gf, ga in [
            (home, m["home_score"], m["away_score"]),
            (away, m["away_score"], m["home_score"]),
        ]:
            if team not in table:
                table[team] = _empty_row(team)
            rec = table[team]
            rec["played"] += 1
            rec["goals_for"] += gf
            rec["goals_against"] += ga
            if gf > ga:
                rec["won"] += 1
                rec["points"] += 3
                rec["form"].append("W")
            elif gf == ga:
                rec["drawn"] += 1
                rec["points"] += 1
                rec["form"].append("D")
            else:
                rec["lost"] += 1
                rec["form"].append("L")

    for team in all_teams:
        if team not in table:
            table[team] = _empty_row(team)

    for rec in table.values():
        rec["goal_difference"] = rec["goals_for"] - rec["goals_against"]
        rec["form"] = "".join(rec["form"][-5:])

    sorted_table = sorted(
        table.values(),
        key=lambda x: (x["points"], x["goal_difference"], x["goals_for"]),
        reverse=True,
    )
    for i, rec in enumerate(sorted_table, 1):
        rec["position"] = i
    return sorted_table


def _empty_row(team: str) -> Dict[str, Any]:
    return {
        "team": team,
        "played": 0,
        "won": 0,
        "drawn": 0,
        "lost": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "points": 0,
        "form": [],
    }


def is_upcoming(m: Any) -> bool:
    """True for scheduled matches dated today or later (SAST). Past-dated
    'scheduled' rows are stale feed entries with no recorded score."""
    if m.get("status") != "scheduled":
        return False
    try:
        return datetime.fromisoformat(m["date"]).date() >= sast_now().date()
    except Exception:
        return False


def sast_now() -> datetime:
    """Return the current time in SAST as a naive datetime."""
    try:
        return datetime.now(_SAST).replace(tzinfo=None)
    except Exception:
        return datetime.now(timezone(timedelta(hours=2))).replace(tzinfo=None)
