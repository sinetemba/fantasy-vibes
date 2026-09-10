"""TheSportsDB source (free demo key, or personal API key)."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils import cached_get, to_sast
from .base import DataSource

logger = logging.getLogger(__name__)

API_BASE = "https://www.thesportsdb.com/api/v1/json"

STATUS_MAP = {
    "FT": "full_time",
    "NS": "scheduled",
    "LIVE": "live",
    "HT": "half_time",
    "POSTP": "postponed",
    "CANC": "cancelled",
}


def _api_key() -> str:
    return os.getenv("THESPORTSDB_API_KEY") or "3"


def _normalize_team(name: str) -> str:
    """Return a short display name from the TheSportsDB team string."""
    if not name:
        return ""
    return name.strip()


class TheSportsDBSource(DataSource):
    """Provider that reads from thesportsdb.com."""

    name = "thesportsdb"

    def is_available(self, league_config: Dict[str, Any]) -> bool:
        return "thesportsdb" in league_config

    def _base_url(self) -> str:
        return f"{API_BASE}/{_api_key()}"

    def get_matches(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        cfg = league_config.get("thesportsdb", {})
        league_id = cfg.get("league_id")
        if not league_id:
            return []
        season_param = season or cfg.get("season", "")
        season_url = f"{self._base_url()}/eventsseason.php?id={league_id}&s={season_param}"
        data = cached_get(season_url, ttl_seconds=3600)

        events = []
        if data:
            events = data.get("events") or []

        # For the current season, also fetch upcoming fixtures which are sometimes
        # not included in the full-season response (e.g. UEFA competitions).
        if season_param and season_param == cfg.get("season"):
            next_url = f"{self._base_url()}/eventsnextleague.php?id={league_id}"
            next_data = cached_get(next_url, ttl_seconds=1800)
            if next_data:
                seen = {e.get("idEvent") for e in events}
                for e in next_data.get("events") or []:
                    if e.get("idEvent") and e.get("idEvent") not in seen:
                        events.append(e)

        matches = []
        for e in events:
            status = STATUS_MAP.get(e.get("strStatus"), "scheduled")
            home = _normalize_team(e.get("strHomeTeam", ""))
            away = _normalize_team(e.get("strAwayTeam", ""))
            home_score = e.get("intHomeScore")
            away_score = e.get("intAwayScore")

            if status == "full_time":
                try:
                    home_score = int(home_score) if home_score is not None else 0
                    away_score = int(away_score) if away_score is not None else 0
                except (ValueError, TypeError):
                    home_score = 0
                    away_score = 0
            else:
                home_score = None
                away_score = None

            ts = e.get("strTimestamp")
            dt = None
            if ts:
                try:
                    dt = to_sast(datetime.fromisoformat(ts.replace("Z", "+00:00")), "UTC")
                except ValueError:
                    pass

            if status == "full_time":
                minutes = "FT"
            elif status == "scheduled" and dt is not None:
                minutes = dt.strftime("%H:%M") + " SAST"
            else:
                minutes = status.replace("_", " ").title()
            matches.append(
                {
                    "match_id": str(e.get("idEvent", f"{home}_{away}")),
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "date": dt.isoformat() if dt else None,
                    "round": f"Matchday {e.get('intRound', '')}",
                    "status": status,
                    "minutes_elapsed": minutes,
                    "venue": e.get("strVenue") or None,
                    "source": self.name,
                }
            )
        matches.sort(key=lambda x: x["date"] or "")
        return matches

    def get_standings(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        cfg = league_config.get("thesportsdb", {})
        league_id = cfg.get("league_id")
        if not league_id:
            return []
        season_param = season or cfg.get("season", "")
        url = f"{self._base_url()}/lookuptable.php?l={league_id}&s={season_param}"
        data = cached_get(url, ttl_seconds=3600)
        if not data:
            return []

        table = data.get("table") or []
        rows = []
        for row in table:
            team = _normalize_team(row.get("strTeam", ""))
            rows.append(
                {
                    "position": int(row.get("intRank", 0)) or None,
                    "team": team,
                    "played": int(row.get("intPlayed", 0)),
                    "won": int(row.get("intWin", 0)),
                    "drawn": int(row.get("intDraw", 0)),
                    "lost": int(row.get("intLoss", 0)),
                    "goals_for": int(row.get("intGoalsFor", 0)),
                    "goals_against": int(row.get("intGoalsAgainst", 0)),
                    "goal_difference": int(row.get("intGoalDifference", 0)),
                    "points": int(row.get("intPoints", 0)),
                    "form": row.get("strForm", ""),
                }
            )
        return rows
