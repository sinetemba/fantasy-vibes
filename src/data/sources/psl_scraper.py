"""Official PSL match-centre scraper for fixtures and results."""

import html as html_lib
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..prediction_engine import _norm_key
from ..utils import CACHE_DIR, cache_ttl, compute_table, to_sast
from .base import DataSource

logger = logging.getLogger(__name__)

PSL_URL = "https://www.psl.co.za/matchcentre"
PSL_GET_URL = "https://www.psl.co.za/MatchCentre/Get/"
REQUEST_TIMEOUT = 20
HTML_CACHE_TTL = 1800  # 30 minutes
PSL_CACHE_FILE = CACHE_DIR / "psl_matchcentre.html"
LOG_CACHE_FILE = CACHE_DIR / "psl_log.html"

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
            return (time.time() - PSL_CACHE_FILE.stat().st_mtime) < cache_ttl(HTML_CACHE_TTL)
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
        # The match-centre page only covers a rolling window of fixtures, so a
        # table computed from it misses earlier results. The site's AJAX log
        # endpoint returns the real season standings; prefer that, with the
        # computed table as fallback.
        rows = self._fetch_log_standings(league_config)
        if not rows:
            return compute_table(self.get_matches(league_config, season))

        # The log has no form column — attach recent form computed from the
        # scraped results window.
        forms = {
            _norm_key(r["team"]): r["form"]
            for r in compute_table(self.get_matches(league_config, season))
        }
        for r in rows:
            r["form"] = forms.get(_norm_key(r["team"]), "")
        return rows

    def _fetch_log_standings(self, league_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        slug = (league_config.get("psl_scraper") or {}).get("log_slug", "betway-premiership")
        try:
            return self._parse_log(self._fetch_log_html(slug))
        except Exception as exc:
            logger.warning(f"PSL log fetch failed: {exc}")
            return []

    def _fetch_log_html(self, slug: str) -> str:
        try:
            if LOG_CACHE_FILE.exists() and LOG_CACHE_FILE.stat().st_mtime > time.time() - cache_ttl(HTML_CACHE_TTL):
                return LOG_CACHE_FILE.read_text(encoding="utf-8")
            r = requests.get(
                PSL_GET_URL,
                params={"leaqueName": slug, "hasLog": "true", "isMatchCentrePage": "true"},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": PSL_URL,
                },
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            html = r.text
            if "table-standings-logs" not in html:
                return ""
            LOG_CACHE_FILE.write_text(html, encoding="utf-8")
            return html
        except Exception as exc:
            logger.warning(f"PSL log request failed: {exc}")
            if LOG_CACHE_FILE.exists():
                return LOG_CACHE_FILE.read_text(encoding="utf-8")
            return ""

    @staticmethod
    def _parse_log(html: str) -> List[Dict[str, Any]]:
        body = re.search(r'<tbody id="LogViewContent">(.*?)</tbody>', html, re.DOTALL)
        if not body:
            return []
        rows: List[Dict[str, Any]] = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body.group(1), re.DOTALL):
            team_m = re.search(r'<h6 class="team-meta__name">([^<]+)</h6>', tr)
            pos_m = re.search(r'<h5 class="team-meta__name">\s*(\d+)', tr)
            if not team_m:
                continue

            def _cell(cls: str) -> int:
                c = re.search(r'<td class="%s">\s*(-?\d+)\s*</td>' % cls, tr)
                return int(c.group(1)) if c else 0

            rows.append({
                "position": int(pos_m.group(1)) if pos_m else len(rows) + 1,
                "team": _clean_team_name(team_m.group(1)) or "",
                "played": _cell("logs-played"),
                "won": _cell("logs-win"),
                "drawn": _cell("logs-draw"),
                "lost": _cell("logs-lost"),
                "goals_for": _cell("logs-goals-for"),
                "goals_against": _cell("logs-goals-against"),
                "goal_difference": _cell("logs-goal-diff"),
                "points": _cell("logs-points"),
                "form": "",
            })
        return rows
