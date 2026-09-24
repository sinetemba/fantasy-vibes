"""BBC Sport source — collated football scores & fixtures (free, no key).

Uses the public JSON feed behind bbc.com/sport/football/scores-fixtures.
The feed covers a rolling window around today, so it is best for current
international windows (friendlies, qualifiers, Nations League) rather than
deep history.
"""

import calendar
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..utils import cached_get, sast_now, to_sast
from .base import DataSource

logger = logging.getLogger(__name__)

BASE_URL = "https://www.bbc.com/wc-data/container/sport-data-scores-fixtures"
FOOTBALL_URN = "urn:bbc:sportsdata:football"

STATUS_MAP = {
    "PreEvent": "scheduled",
    "MidEvent": "live",
    "PostEvent": "full_time",
    "Postponed": "postponed",
    "Cancelled": "cancelled",
}

# The collated feed only serves windows that don't straddle a month
# boundary: the current month can start at most ~7 days before today, and
# other windows must be inside a single calendar month.
DEFAULT_DAYS_BACK = 6
DEFAULT_DAYS_AHEAD = 150


def _event_day(e: Dict[str, Any]) -> Optional[date]:
    try:
        return datetime.fromisoformat(
            (e.get("startDateTime") or "").replace("Z", "+00:00")
        ).date()
    except ValueError:
        return None


def _dedupe_events(
    events: List[Tuple[Dict[str, Any], str]],
) -> List[Tuple[Dict[str, Any], str]]:
    """Collapse duplicate feed entries for the same fixture.

    The feed sometimes emits a fixture more than once — e.g. a date-only
    "kick-off time to be confirmed" placeholder next to the real timed
    event, occasionally with a date a day or two off. Treat same-pairing
    events as duplicates when they share a round (a pairing meets once
    per round) or land within a day of each other, and keep the entry
    with the more certain kickoff time.
    """

    def pairing(e: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        home = e.get("home") or {}
        away = e.get("away") or {}
        return (
            (e.get("tournament") or {}).get("id") or e.get("tournamentId"),
            home.get("id") or home.get("fullName"),
            away.get("id") or away.get("fullName"),
        )

    def round_id(e: Dict[str, Any]) -> Any:
        r = e.get("round") or {}
        return r.get("id") or r.get("name")

    def timed(e: Dict[str, Any]) -> bool:
        return bool((e.get("time") or {}).get("timeCertainty")) or "T" in (
            e.get("startDateTime") or ""
        )

    def is_dupe(e: Dict[str, Any], k: Dict[str, Any]) -> bool:
        if pairing(k) != pairing(e):
            return False
        day, kday = _event_day(e), _event_day(k)
        if day is None or kday is None or abs((day - kday).days) <= 1:
            return True
        # Same pairing in the same round is also a dupe when at least one
        # entry is an unconfirmed-time placeholder — two timed entries in
        # one round may be genuinely distinct fixtures.
        er, kr = round_id(e), round_id(k)
        return er is not None and er == kr and not (timed(e) and timed(k))

    kept: List[Tuple[Dict[str, Any], str]] = []
    for e, label in events:
        for i, (k, _) in enumerate(kept):
            if not is_dupe(e, k):
                continue
            if timed(e) and not timed(k):
                kept[i] = (e, label)
            break
        else:
            kept.append((e, label))
    return kept


def _date_windows(back: int, ahead: int) -> List[Tuple[str, str]]:
    """(start, end) ISO date pairs covering [~today-7, today+ahead]."""
    today = sast_now().date()
    month_last = calendar.monthrange(today.year, today.month)[1]
    month_start = date(today.year, today.month, 1)
    month_end = date(today.year, today.month, month_last)

    windows = [
        (
            max(today - timedelta(days=min(back, 6)), month_start).isoformat(),
            month_end.isoformat(),
        )
    ]
    horizon = today + timedelta(days=ahead)
    y, m = (today.year, today.month + 1) if today.month < 12 else (today.year + 1, 1)
    while True:
        start = date(y, m, 1)
        if start > horizon:
            break
        windows.append(
            (start.isoformat(), date(y, m, calendar.monthrange(y, m)[1]).isoformat())
        )
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    return windows


class BBCSource(DataSource):
    """Provider that reads BBC Sport's collated football scores feed."""

    name = "bbc"

    def is_available(self, league_config: Dict[str, Any]) -> bool:
        return "bbc" in league_config

    def _fetch(self, start: str, end: str) -> List[Dict[str, Any]]:
        today = sast_now().date().isoformat()
        url = (
            f"{BASE_URL}?urn={FOOTBALL_URN}"
            f"&selectedStartDate={start}&selectedEndDate={end}&todayDate={today}"
        )
        data = cached_get(url, ttl_seconds=900)
        if not data:
            return []
        return data.get("eventGroups") or []

    def get_matches(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        cfg = league_config.get("bbc", {})
        tournaments = set(cfg.get("tournaments") or [])
        if not tournaments:
            return []

        back = int(cfg.get("days_back", DEFAULT_DAYS_BACK))
        ahead = int(cfg.get("days_ahead", DEFAULT_DAYS_AHEAD))

        windows = _date_windows(back, ahead)
        with ThreadPoolExecutor(max_workers=min(6, len(windows) or 1)) as pool:
            window_groups = list(pool.map(lambda w: self._fetch(*w), windows))

        events: List[Tuple[Dict[str, Any], str]] = []
        for groups in window_groups:
            for group in groups:
                for sub in group.get("secondaryGroups") or []:
                    round_label = sub.get("displayLabel") or ""
                    for e in sub.get("events") or []:
                        if (e.get("tournament") or {}).get("name") not in tournaments:
                            continue
                        events.append((e, round_label))

        matches = []
        for e, round_label in _dedupe_events(events):
            m = self._to_match(e, round_label)
            if m:
                matches.append(m)

        matches.sort(key=lambda x: x["date"] or "")
        return matches

    def _to_match(
        self, e: Dict[str, Any], round_label: str
    ) -> Optional[Dict[str, Any]]:
        home = (e.get("home") or {}).get("fullName", "").strip()
        away = (e.get("away") or {}).get("fullName", "").strip()
        if not home or not away:
            return None

        period = (e.get("periodLabel") or {}).get("value", "")
        status = STATUS_MAP.get(e.get("status"), "scheduled")
        if status == "live" and period == "HT":
            status = "half_time"

        home_score = away_score = None
        if status in ("full_time", "live", "half_time"):
            try:
                home_score = int((e["home"].get("score")))
                away_score = int((e["away"].get("score")))
            except (TypeError, ValueError, KeyError):
                home_score = away_score = None
                if status == "full_time":
                    status = "scheduled"

        dt = None
        start = e.get("startDateTime")
        if start:
            try:
                dt = to_sast(datetime.fromisoformat(start.replace("Z", "+00:00")), "UTC")
            except ValueError:
                pass

        # Date-only feed entries have no confirmed kickoff — their datetime
        # is just midnight UTC, so rendering it as "02:00 SAST" is a lie.
        time_field = e.get("time") or {}
        if "timeCertainty" in time_field:
            time_certain = bool(time_field["timeCertainty"])
        else:
            time_certain = "T" in (e.get("startDateTime") or "")
        if period:
            minutes = period
        elif status == "scheduled" and dt is not None and time_certain:
            minutes = dt.strftime("%H:%M") + " SAST"
        elif status == "scheduled" and not time_certain:
            minutes = "TBC"
        else:
            minutes = status.replace("_", " ").title()

        return {
            "match_id": e.get("id") or f"bbc_{home}_{away}_{e.get('startDateTime')}",
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "date": dt.isoformat() if dt else None,
            "round": round_label,
            "status": status,
            "minutes_elapsed": minutes,
            "time_confirmed": time_certain,
            "venue": None,
            "source": self.name,
        }

    def get_standings(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return []
