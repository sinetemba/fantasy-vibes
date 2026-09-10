"""FixtureDownload.com source (free public fixture list)."""

import html
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils import cached_get, to_sast
from .base import DataSource

logger = logging.getLogger(__name__)

BASE_URL = "https://fixturedownload.com/view/json"


class FixtureDownloadSource(DataSource):
    """Provider that reads fixture lists from fixturedownload.com."""

    name = "fixturedownload"

    def is_available(self, league_config: Dict[str, Any]) -> bool:
        return "fixturedownload" in league_config

    def _fetch(self, slug: str, season: str) -> List[Dict[str, Any]]:
        url = f"{BASE_URL}/{slug}-{season}"
        text = cached_get(url, ttl_seconds=3600, raw=True)
        if not text:
            return []
        m = re.search(r"<textarea[^>]*>(.*?)</textarea>", text, re.DOTALL)
        if not m:
            return []
        raw = html.unescape(m.group(1))
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse %s: %s", url, exc)
            return []

    def get_matches(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        cfg = league_config.get("fixturedownload", {})
        slug = cfg.get("slug")
        if not slug:
            return []
        season_param = season or cfg.get("season", "")
        if not season_param:
            return []

        events = self._fetch(slug, season_param)
        if not events:
            return []

        matches: List[Dict[str, Any]] = []
        for i, e in enumerate(events):
            home = e.get("HomeTeam", "").strip()
            away = e.get("AwayTeam", "").strip()
            if not home or not away:
                continue

            home_score = e.get("HomeTeamScore")
            away_score = e.get("AwayTeamScore")
            if home_score is not None and away_score is not None:
                try:
                    home_score = int(home_score)
                    away_score = int(away_score)
                    status = "full_time"
                    minutes = "FT"
                except (ValueError, TypeError):
                    home_score = None
                    away_score = None
                    status = "scheduled"
                    minutes = status.replace("_", " ").title()
            else:
                home_score = None
                away_score = None
                status = "scheduled"
                minutes = status.replace("_", " ").title()

            dt = None
            date_utc = e.get("DateUtc")
            if date_utc:
                try:
                    dt_utc = datetime.strptime(date_utc, "%Y-%m-%d %H:%M:%SZ")
                    dt = to_sast(dt_utc, "UTC")
                except ValueError:
                    pass

            round_number = e.get("RoundNumber")
            group = e.get("Group")
            if group:
                round_label = f"Group {group}"
                if round_number:
                    round_label += f" - Round {round_number}"
            elif round_number:
                round_label = f"Matchday {round_number}"
            else:
                round_label = ""

            match_id = f"fd_{e.get('MatchNumber', i)}_{home}_{away}"
            matches.append(
                {
                    "match_id": match_id,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "date": dt.isoformat() if dt else None,
                    "round": round_label,
                    "status": status,
                    "minutes_elapsed": minutes,
                    "venue": e.get("Location") or None,
                    "source": self.name,
                }
            )

        matches.sort(key=lambda x: x["date"] or "")
        return matches

    def get_standings(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return []
