"""Football-Data.org API source (requires a free API key)."""

import logging
import os
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils import cached_get, sast_now, to_sast
from .base import DataSource

logger = logging.getLogger(__name__)

API_BASE = "https://api.football-data.org/v4"
REQUEST_TIMEOUT = 15

STATUS_MAP = {
    "SCHEDULED": "scheduled",
    "TIMED": "scheduled",
    "IN_PLAY": "live",
    "LIVE": "live",
    "PAUSED": "half_time",
    "FINISHED": "full_time",
    "AWARDED": "full_time",
    "POSTPONED": "postponed",
    "SUSPENDED": "cancelled",
    "CANCELLED": "cancelled",
}


def _load_api_key() -> Optional[str]:
    key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not key and "__football_data_api_key" in os.environ:
        # defensive guard in case dotenv loads with a different shape
        pass
    return key or None


def _headers() -> Dict[str, str]:
    return {"X-Auth-Token": _load_api_key() or ""}


class FootballDataSource(DataSource):
    """Provider that reads from football-data.org."""

    name = "football-data.org"

    def is_available(self, league_config: Dict[str, Any]) -> bool:
        return bool(_load_api_key()) and "football_data" in league_config

    def _api_get(
        self,
        path: str,
        params: Optional[Dict] = None,
        ttl_seconds: int = 300,
    ) -> Optional[Dict[str, Any]]:
        if not _load_api_key():
            return None
        url = f"{API_BASE}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = cached_get(url, headers=_headers(), ttl_seconds=ttl_seconds, timeout=REQUEST_TIMEOUT)
        if data is None:
            logger.warning("Football-Data.org returned no data (rate-limited, key rejected, or unavailable)")
        return data

    def get_matches(
        self,
        league_config: Dict[str, Any],
        season: Optional[str] = None,
        status_filter: Optional[str] = None,
        ttl_seconds: int = 300,
    ) -> List[Dict[str, Any]]:
        cfg = league_config.get("football_data", {})
        code = cfg.get("code", "PL")
        season_param = season or cfg.get("season", "")
        params: Dict[str, str] = {}
        if season_param:
            params["season"] = season_param
        if status_filter:
            params["status"] = status_filter
        if not params:
            params = None  # type: ignore[assignment]
        data = self._api_get(f"/competitions/{code}/matches", params, ttl_seconds=ttl_seconds)
        if not data:
            return []

        matches = []
        for m in data.get("matches", []):
            status = STATUS_MAP.get(m.get("status", "SCHEDULED"), "scheduled")
            home_team = (m.get("homeTeam") or {}).get("name", "TBD")
            away_team = (m.get("awayTeam") or {}).get("name", "TBD")
            score = m.get("score") or {}
            ft = score.get("fullTime") or {}
            home_score = ft.get("home")
            away_score = ft.get("away")
            if status == "full_time" and home_score is None:
                home_score = 0
                away_score = 0

            utc_date = m.get("utcDate")
            dt = None
            if utc_date:
                try:
                    dt = to_sast(datetime.fromisoformat(utc_date.replace("Z", "+00:00")), "UTC")
                except ValueError:
                    pass

            minute = None
            if status == "live":
                minute = m.get("minute")
            if status == "live" and minute is not None:
                minutes_elapsed = f"{minute}'"
            elif status == "full_time":
                minutes_elapsed = "FT"
            elif status == "half_time":
                minutes_elapsed = "HT"
            elif status == "scheduled" and dt is not None:
                minutes_elapsed = dt.strftime("%H:%M") + " SAST"
            else:
                minutes_elapsed = status.replace("_", " ").title()

            matches.append(
                {
                    "match_id": str(m.get("id", f"{home_team}_{away_team}")),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "date": dt.isoformat() if dt else None,
                    "round": f"Matchday {m.get('matchday', '')}",
                    "status": status,
                    "minutes_elapsed": minutes_elapsed,
                    "venue": (m.get("venue") or "TBD"),
                    "source": self.name,
                }
            )

        matches.sort(key=lambda x: x["date"] or "")
        return matches

    def get_live_matches(self, league_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch only currently live matches with a short cache so scores update."""
        return self.get_matches(league_config, status="LIVE", ttl_seconds=15)

    def get_standings(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        cfg = league_config.get("football_data", {})
        code = cfg.get("code", "PL")
        season_param = season or cfg.get("season", "")
        params = {"season": season_param} if season_param else None
        data = self._api_get(f"/competitions/{code}/standings", params)
        if not data or "standings" not in data:
            return []

        standings = data["standings"]
        if not standings or not isinstance(standings, list):
            return []
        table = standings[0].get("table", [])

        rows = []
        for row in table:
            team = (row.get("team") or {}).get("name", "")
            rows.append(
                {
                    "position": row.get("position"),
                    "team": team,
                    "played": row.get("playedGames", 0),
                    "won": row.get("won", 0),
                    "drawn": row.get("draw", 0),
                    "lost": row.get("lost", 0),
                    "goals_for": row.get("goalsFor", 0),
                    "goals_against": row.get("goalsAgainst", 0),
                    "goal_difference": row.get("goalDifference", 0),
                    "points": row.get("points", 0),
                    "form": row.get("form", ""),
                }
            )
        return rows
