"""Official PSL match-centre scraper for fixtures and results."""

import html as html_lib
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..utils import CACHE_DIR, to_sast
from .base import DataSource

logger = logging.getLogger(__name__)

PSL_URL = "https://www.psl.co.za/matchcentre"
REQUEST_TIMEOUT = 20
HTML_CACHE_TTL = 1800  # 30 minutes
PSL_CACHE_FILE = CACHE_DIR / "psl_matchcentre.html"

# Known bad labels on the PSL site where a stadium name leaks into the team field,
# plus name variants to align with the TheSportsDB spelling.
_TEAM_NAME_FIXES = {
    "Orlando Amstel Arena": "Orlando Pirates",
    "AmaZulu": "Amazulu",
}


def _text(tag_match: Optional[re.Match]) -> Optional[str]:
    if not tag_match:
        return None
    return html_lib.unescape(tag_match.group(1).strip())


def _clean_team_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    name = " ".join(raw.split())
    # Strip trailing " FC" to align with other sources.
    if name.endswith(" FC"):
        name = name[:-3].strip()
    name = _TEAM_NAME_FIXES.get(name, name)
    return name


class PSLScraperSource(DataSource):
    """Provider that scrapes the official PSL match centre."""

    name = "psl-scraper"

    def is_available(self, league_config: Dict[str, Any]) -> bool:
        return "psl_scraper" in league_config

    def _is_cache_valid(self) -> bool:
        if not PSL_CACHE_FILE.exists():
            return False
        try:
            return (time.time() - PSL_CACHE_FILE.stat().st_mtime) < HTML_CACHE_TTL
        except Exception:
            return False

    def _read_cache(self) -> Optional[str]:
        try:
            return PSL_CACHE_FILE.read_text(encoding="utf-8")
        except Exception:
            return None

    def _write_cache(self, text: str) -> None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            PSL_CACHE_FILE.write_text(text, encoding="utf-8")
        except Exception:
            pass

    def _fetch_html(self) -> Optional[str]:
        if self._is_cache_valid():
            cached = self._read_cache()
            if cached:
                return cached

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(PSL_URL, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            text = response.text
            self._write_cache(text)
            return text
        except Exception as exc:
            logger.warning(f"Failed to fetch PSL match centre: {exc}")

        # Serve stale cache if the live request failed.
        stale = self._read_cache()
        if stale:
            return stale
        return None

    def _parse_team(self, section_html: str, team_class: str) -> Optional[str]:
        pattern = rf'<td class="{team_class}">.*?<h6[^>]*class="team-meta__name">([^<]+)</h6>'
        match = re.search(pattern, section_html, re.DOTALL)
        return _clean_team_name(_text(match))

    def _parse_score(self, section_html: str) -> Optional[tuple]:
        match = re.search(
            r'<td class="results-score">\s*<span[^>]*>\s*(\d+)\s*-\s*(\d+)\s*</span>',
            section_html,
            re.DOTALL,
        )
        if match:
            try:
                return int(match.group(1)), int(match.group(2))
            except (ValueError, TypeError):
                return None
        return None

    def _parse_time(self, section_html: str, date_code: str) -> Optional[datetime]:
        footer_match = re.search(
            r'<td[^>]*class="fixtures-footer-block">([^<]+)</td>',
            section_html,
            re.DOTALL,
        )
        time_str = None
        if footer_match:
            footer_text = _text(footer_match)
            if footer_text:
                time_match = re.search(r'(\d{1,2}:\d{2})', footer_text)
                if time_match:
                    time_str = time_match.group(1)
        try:
            if time_str:
                dt = datetime.strptime(f"{date_code} {time_str}", "%d%b%Y %H:%M")
            else:
                dt = datetime.strptime(date_code, "%d%b%Y")
            return to_sast(dt, "Africa/Johannesburg")
        except ValueError:
            return None

    def get_matches(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            page = self._fetch_html()
            if not page:
                return []

            matches = []
            # Each match sits in its own <tbody name="fixtures_13Sep2026"> or "results_..."
            for match in re.finditer(
                r'<tbody[^>]*name="(fixtures|results)_(\d+[A-Za-z]{3}\d{4})"[^>]*>(.*?)</tbody>',
                page,
                re.DOTALL,
            ):
                section = match.group(1)
                date_code = match.group(2)
                body = match.group(3)

                home = self._parse_team(
                    body, "fixtures-team1" if section == "fixtures" else "results-team1"
                )
                away = self._parse_team(
                    body, "fixtures-team2" if section == "fixtures" else "results-team2"
                )
                if not home or not away:
                    continue

                dt = self._parse_time(body, date_code)
                try:
                    naive_date = datetime.strptime(date_code, "%d%b%Y").date()
                except ValueError:
                    naive_date = None

                if section == "results":
                    score = self._parse_score(body)
                    status = "full_time"
                    home_score = score[0] if score else None
                    away_score = score[1] if score else None
                    minutes = "FT"
                else:
                    status = "scheduled"
                    home_score = None
                    away_score = None
                    minutes = dt.strftime("%H:%M") + " SAST" if dt else "TBD"

                match_id = f"{home}_{away}_{naive_date.isoformat() if naive_date else ''}".replace(
                    " ", "_"
                )
                matches.append(
                    {
                        "match_id": match_id,
                        "home_team": home,
                        "away_team": away,
                        "home_score": home_score,
                        "away_score": away_score,
                        "date": dt.isoformat() if dt else None,
                        "round": "",
                        "status": status,
                        "minutes_elapsed": minutes,
                        "venue": None,
                        "source": self.name,
                    }
                )

            matches.sort(key=lambda x: x["date"] or "")
            logger.info(f"PSL scraper parsed {len(matches)} fixtures/results")
            return matches
        except Exception as exc:
            logger.warning(f"PSL scraper parse failed, falling back: {exc}")
            return []

    def get_standings(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        matches = self.get_matches(league_config, season)
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

            for team, gf, ga in [(home, m["home_score"], m["away_score"]), (away, m["away_score"], m["home_score"])]:
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
                elif gf == ga:
                    rec["drawn"] += 1
                    rec["points"] += 1
                    rec["form"].append("D")
                else:
                    rec["lost"] += 1
                    rec["form"].append("L")

        # Include every team, even those with no recorded result yet.
        for team in all_teams:
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
