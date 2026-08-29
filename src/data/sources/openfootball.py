"""Openfootball public JSON data source (no API key required)."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..utils import cached_get, sast_now, to_sast
from .base import DataSource

logger = logging.getLogger(__name__)

OPENFOOTBALL_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"

STATUS_SCHEDULED = "scheduled"
STATUS_LIVE = "live"
STATUS_FULL_TIME = "full_time"
STATUS_POSTPONED = "postponed"


def _parse_score(score_obj: Any) -> Optional[tuple]:
    """Extract (home, away) full-time score from openfootball score field."""
    if score_obj is None:
        return None
    if isinstance(score_obj, dict):
        ft = score_obj.get("ft")
        if isinstance(ft, (list, tuple)) and len(ft) == 2:
            try:
                return int(ft[0]), int(ft[1])
            except (ValueError, TypeError):
                return None
    if isinstance(score_obj, (list, tuple)) and len(score_obj) == 2:
        try:
            return int(score_obj[0]), int(score_obj[1])
        except (ValueError, TypeError):
            return None
    return None


def _to_naive_datetime(date_str: str, time_str: Optional[str] = None) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if time_str:
            try:
                t = datetime.strptime(time_str, "%H:%M").time()
                dt = datetime.combine(dt.date(), t)
            except ValueError:
                pass
        return dt
    except ValueError:
        return None


def _derive_status(
    dt: Optional[datetime], score: Optional[tuple]
) -> str:
    now = sast_now()
    if score is not None:
        return STATUS_FULL_TIME
    if dt is None:
        return STATUS_SCHEDULED
    if now < dt:
        return STATUS_SCHEDULED
    if dt <= now < dt + timedelta(hours=2, minutes=30):
        return STATUS_LIVE
    # Past match with no score data — treat as scheduled/postponed
    if now > dt + timedelta(days=1):
        return STATUS_POSTPONED
    return STATUS_SCHEDULED


def _format_minutes(status: str, dt: Optional[datetime]) -> str:
    if status == STATUS_FULL_TIME:
        return "FT"
    if status == STATUS_LIVE:
        if dt:
            minutes = int((sast_now() - dt).total_seconds() / 60)
            return f"{minutes}'" if minutes <= 120 else "ET"
        return "LIVE"
    if status == STATUS_POSTPONED:
        return "Postponed"
    if dt:
        return dt.strftime("%H:%M") + " SAST"
    return "TBD"


class OpenfootballSource(DataSource):
    """Provider that reads from the openfootball/football.json GitHub repository."""

    name = "openfootball"

    def is_available(self, league_config: Dict[str, Any]) -> bool:
        return "openfootball" in league_config

    def _fetch(self, league_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cfg = league_config.get("openfootball", {})
        season = cfg.get("season", "2025-26")
        path = cfg.get("path", "en.1.json")
        url = f"{OPENFOOTBALL_BASE}/{season}/{path}"
        return cached_get(url, ttl_seconds=86400)

    def get_matches(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        data = self._fetch(league_config)
        if not data:
            return []

        cfg = league_config.get("openfootball", {})
        source_tz = cfg.get("timezone", "Europe/London")
        matches = []
        for m in data.get("matches", []):
            team1 = m.get("team1", "")
            team2 = m.get("team2", "")
            date = m.get("date", "")
            time = m.get("time")
            score = _parse_score(m.get("score"))
            dt = to_sast(_to_naive_datetime(date, time), source_tz)
            status = _derive_status(dt, score)

            match = {
                "match_id": f"{team1}_{team2}_{date}".replace(" ", "_"),
                "home_team": team1,
                "away_team": team2,
                "home_score": score[0] if score else None,
                "away_score": score[1] if score else None,
                "date": dt.isoformat() if dt else None,
                "round": m.get("round", ""),
                "status": status,
                "minutes_elapsed": _format_minutes(status, dt),
                "venue": None,
                "source": self.name,
            }
            matches.append(match)

        matches.sort(key=lambda x: x["date"] or "")
        return matches

    def get_standings(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        matches = self.get_matches(league_config, season)
        table = {}

        for m in matches:
            if m["status"] != STATUS_FULL_TIME:
                continue
            home = m["home_team"]
            away = m["away_team"]
            hg = m["home_score"]
            ag = m["away_score"]
            if hg is None or ag is None:
                continue

            for team, gf, ga, is_home in [(home, hg, ag, True), (away, ag, hg, False)]:
                if team not in table:
                    table[team] = {
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
                rec = table[team]
                rec["played"] += 1
                rec["goals_for"] += gf
                rec["goals_against"] += ga

                if gf > ga:
                    rec["won"] += 1
                    rec["points"] += 3
                    rec["form"].append("W")
                elif gf == ag:
                    rec["drawn"] += 1
                    rec["points"] += 1
                    rec["form"].append("D")
                else:
                    rec["lost"] += 1
                    rec["form"].append("L")

        for rec in table.values():
            rec["goal_difference"] = rec["goals_for"] - rec["goals_against"]
            rec["form"] = "".join(rec["form"][-5:][::-1])

        sorted_table = sorted(
            table.values(),
            key=lambda x: (x["points"], x["goal_difference"], x["goals_for"]),
            reverse=True,
        )
        for i, rec in enumerate(sorted_table, 1):
            rec["position"] = i
        return sorted_table
