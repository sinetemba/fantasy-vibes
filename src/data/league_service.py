"""High-level service that loads league data and makes predictions."""

import difflib
import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

from .prediction_engine import (
    PredictionEngine,
    _norm_key,
    _TEAM_KEY_ALIASES,
    fixtures_equivalent,
)
from . import prediction_log
from .utils import compute_table, sast_now
from .sources.bbc import BBCSource
from .sources.composite import CompositeDataSource
from .sources.fixture_download import FixtureDownloadSource
from .sources.football_data import FootballDataSource
from .sources.openfootball import OpenfootballSource
from .sources.psl_scraper import PSLScraperSource
from .sources.thesportsdb import TheSportsDBSource

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEAGUES_PATH = PROJECT_ROOT / "data" / "leagues.json"
MODEL_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
MODEL_TTL = 1800  # seconds
GROUP_CACHE_TTL = 300  # seconds — group pages refetch at most this often
NATIONAL_RANKINGS_PATH = PROJECT_ROOT / "data" / "national_rankings.json"
TEAM_VALUES_PATH = PROJECT_ROOT / "data" / "team_values.json"
# Elo points per natural-log unit of squad market value above/below the
# file's geometric mean — a club worth 2.7x the median starts ~+90.
VALUE_ELO_PER_LN = 90.0

# Matches youth/reserve/women's sides in international feeds,
# e.g. "Portugal U17", "Germany U21", "England Women".
_NON_SENIOR_TEAM = re.compile(r"\bU-?\d{2}\b|\bWomen\b|\bLadies\b", re.IGNORECASE)

# Feed/display name -> name used in national_rankings.json.
_NATIONAL_ALIASES = {
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde": "Cabo Verde",
    "Czech Republic": "Czechia",
    "Democratic Republic of the Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Korea South": "South Korea",
    "North Korea": "Korea DPR",
    "Russia": "Russia",
    "Swaziland": "Eswatini",
    "The Gambia": "Gambia",
    "Ireland": "Republic of Ireland",
    "Turkey": "Türkiye",
}

_national_rankings_cache: Optional[Dict[str, float]] = None
_team_values_cache: Optional[Dict[str, float]] = None


def _load_national_rankings() -> Dict[str, float]:
    """National-team strength priors used to seed Elo for internationals."""
    global _national_rankings_cache
    if _national_rankings_cache is None:
        _national_rankings_cache = {}
        try:
            with open(NATIONAL_RANKINGS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if not k.startswith("_") and isinstance(v, (int, float)):
                    _national_rankings_cache[k] = float(v)
        except Exception as exc:
            logger.warning(f"Could not load national rankings: {exc}")
    return _national_rankings_cache


def _load_team_values() -> Dict[str, float]:
    """Squad market values (EUR millions) used to seed Elo for clubs."""
    global _team_values_cache
    if _team_values_cache is None:
        _team_values_cache = {}
        try:
            with open(TEAM_VALUES_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if not k.startswith("_") and isinstance(v, (int, float)):
                    _team_values_cache[k] = float(v)
        except Exception as exc:
            logger.warning(f"Could not load team values: {exc}")
    return _team_values_cache


# Competition weighting for match importance — competitive international
# fixtures outrank friendlies at equal team strength; CL/EL get a bonus so
# club games surface on non-international days.
_COMP_BONUS = {"CL": 150, "EL": 80, "INTF": 0}
_DEFAULT_INTL_BONUS = 100
_leagues_registry_cache: Optional[Dict[str, Any]] = None


def _leagues_registry() -> Dict[str, Any]:
    global _leagues_registry_cache
    if _leagues_registry_cache is None:
        try:
            with open(LEAGUES_PATH, "r", encoding="utf-8") as f:
                _leagues_registry_cache = json.load(f)
        except Exception:
            _leagues_registry_cache = {}
    return _leagues_registry_cache


def match_importance(m: Dict[str, Any]) -> float:
    """Marquee score for a fixture: combined national-team rating plus a
    competition bonus. Returns 0 for unrated domestic club games."""
    ranks = _load_national_rankings()
    base = (
        ranks.get(
            _NATIONAL_ALIASES.get(m.get("home_team", ""), m.get("home_team", "")), 0.0
        )
        + ranks.get(
            _NATIONAL_ALIASES.get(m.get("away_team", ""), m.get("away_team", "")), 0.0
        )
    )
    code = m.get("league_code") or m.get("competition") or ""
    if code in _COMP_BONUS:
        return base + _COMP_BONUS[code]
    if _leagues_registry().get(code, {}).get("group") == "international":
        return base + _DEFAULT_INTL_BONUS
    return base


def dedupe_matches(
    matches: List[Dict[str, Any]],
    prefer: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """Drop duplicate rows for the same fixture coming from different feeds
    (current + historical + related pools overlap). Buckets candidates by
    match day ±1 so fixtures_equivalent only compares near neighbours —
    O(n) per bucket instead of O(n²) over the whole pool.

    By default the earliest-dated row wins; `prefer(candidate, kept)`
    returning True swaps a later duplicate in for the row already kept."""
    by_day: Dict[str, List[int]] = {}
    undated: List[int] = []
    unique: List[Dict[str, Any]] = []
    for m in sorted(matches, key=lambda x: x.get("date") or ""):
        day = (m.get("date") or "")[:10]
        candidates: List[int] = list(undated)
        if day:
            try:
                d = date.fromisoformat(day)
                for delta in (-1, 0, 1):
                    candidates.extend(
                        by_day.get((d + timedelta(days=delta)).isoformat(), [])
                    )
            except ValueError:
                candidates = list(range(len(unique)))
        dup = next(
            (i for i in candidates if fixtures_equivalent(m, unique[i])), None
        )
        if dup is not None:
            if prefer and prefer(m, unique[dup]):
                unique[dup] = m
            continue
        (by_day.setdefault(day, []) if day else undated).append(len(unique))
        unique.append(m)
    return unique


def _senior_teams_only(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        m
        for m in matches
        if not (
            _NON_SENIOR_TEAM.search(m.get("home_team") or "")
            or _NON_SENIOR_TEAM.search(m.get("away_team") or "")
        )
    ]


class LeagueService:
    """Loads fixtures, tables and provides match predictions for a league."""

    def __init__(self, league_code: str = "PL"):
        self._leagues = self._load_leagues()
        self._match_sources = [
            FixtureDownloadSource(),
            BBCSource(),
            FootballDataSource(),
            OpenfootballSource(),
            PSLScraperSource(),
            TheSportsDBSource(),
        ]
        self._standings_sources = [
            FootballDataSource(),
            OpenfootballSource(),
            PSLScraperSource(),
            TheSportsDBSource(),
            FixtureDownloadSource(),
        ]
        self._source = CompositeDataSource(self._match_sources, self._standings_sources)
        self._current_code = None
        self._current_league = None
        self.matches: List[Dict[str, Any]] = []
        self._historical_matches: List[Dict[str, Any]] = []
        self._related: Dict[str, Dict[str, Any]] = {}
        self._team_contexts: Dict[str, Dict[str, Any]] = {}
        self._group_match_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._form_index: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self.standings: List[Dict[str, Any]] = []
        self._match_source_name = None
        self._standings_source_name = None
        self._engine = PredictionEngine()
        # Services are shared across sessions and threads. Writers (warm-up,
        # background refresh, Live refresh, Retrain) serialise on this lock
        # and build new state in locals before swapping it in, so readers
        # take no lock and never see a half-updated service.
        self._lock = threading.RLock()
        self.set_league(league_code)

    def _load_leagues(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(LEAGUES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"Could not load {LEAGUES_PATH}: {exc}")
            return {}

    def set_league(self, code: str):
        if code not in self._leagues:
            raise ValueError(f"Unknown league code: {code}")
        self._current_code = code
        self._current_league = self._leagues[code]
        self._refresh()

    def _refresh(self, refit: bool = False):
        """Full reload: matches, standings, history, related leagues and the
        model. `refit` retrains even when a fresh persisted model exists."""
        with self._lock:
            league = self._current_league

            # Matches and standings use disjoint provider lists, so they
            # fetch in parallel. Historical pulls depend on which source won
            # the match cascade, so they start once matches land and overlap
            # the standings fetch.
            with ThreadPoolExecutor(max_workers=3) as pool:
                matches_fut = pool.submit(self._source.get_matches_with_source, league)
                standings_fut = pool.submit(self._source.get_standings_with_source, league)
                matches, match_source = matches_fut.result()
                hist_fut = pool.submit(self._fetch_historical, match_source)
                standings, standings_source = standings_fut.result()
                historical = hist_fut.result()

            if league.get("senior_only"):
                matches = _senior_teams_only(matches)
            # Fallback to a live table computed from results if no source provides standings.
            if not standings and matches and not league.get("no_table"):
                standings_source = "computed"

            # Tag every match with its competition so the engine can track
            # per-tournament form (e.g. domestic league vs Champions League).
            for m in matches:
                m.setdefault("competition", self._current_code)
            for m in historical:
                m.setdefault("competition", self._current_code)

            # Pull related competitions (e.g. the domestic leagues of CL/EL
            # clubs) so predictions can use each team's own log position and
            # other-tournament form.
            related = self._load_related()

            engine = None
            if not refit and self._model_is_fresh():
                try:
                    engine = self._load_model()
                    self._seed_elo(engine)
                except Exception as exc:
                    logger.warning(f"Persisted model invalid, retraining: {exc}")
                    engine = None
            if engine is None:
                engine = self._train_engine(matches, historical, related, match_source)
            contexts = self._build_team_contexts(engine, matches, standings, related)

            self.matches = matches
            self.standings = standings
            self._historical_matches = historical
            self._related = related
            self._match_source_name = match_source
            self._standings_source_name = standings_source
            self._engine = engine
            self._team_contexts = contexts
            self._form_index = None
            self._group_match_cache.clear()

    def _fetch_historical(self, match_source: Optional[str]) -> List[Dict[str, Any]]:
        """Pull configured historical/archived seasons as extra training data."""
        historical: List[Dict[str, Any]] = []

        # Pull openfootball history as extra training data when the live source is different
        if "openfootball" in self._current_league and match_source != "openfootball":
            try:
                historical = OpenfootballSource().get_matches(self._current_league)
            except Exception:
                historical = []

        # Pull football-data historical seasons as extra training data
        if "football_data" in self._current_league:
            for hist_season in self._current_league["football_data"].get("historical", []):
                try:
                    historical.extend(
                        FootballDataSource().get_matches(self._current_league, season=hist_season)
                    )
                except Exception:
                    pass

        # Pull TheSportsDB historical seasons as extra training data
        if "thesportsdb" in self._current_league:
            for hist_season in self._current_league["thesportsdb"].get("historical", []):
                try:
                    historical.extend(
                        TheSportsDBSource().get_matches(self._current_league, season=hist_season)
                    )
                except Exception:
                    pass

        # Pull FixtureDownload historical seasons as extra training data
        if "fixturedownload" in self._current_league:
            for hist_season in self._current_league["fixturedownload"].get("historical", []):
                try:
                    historical.extend(
                        FixtureDownloadSource().get_matches(self._current_league, season=hist_season)
                    )
                except Exception:
                    pass
        return historical

    def _load_related(self) -> Dict[str, Dict[str, Any]]:
        """Fetch matches and standings for leagues listed in `related`."""
        entries = [
            (code, self._leagues[code])
            for code in self._current_league.get("related", [])
            if code in self._leagues
        ]

        def _fetch(code: str, cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            try:
                matches = self._source.get_matches(cfg)
            except Exception as exc:
                logger.warning(f"Related league {code} matches failed: {exc}")
                matches = []
            if cfg.get("senior_only"):
                matches = _senior_teams_only(matches)
            for m in matches:
                m["competition"] = code
            try:
                table = self._source.get_standings(cfg)
            except Exception as exc:
                logger.warning(f"Related league {code} standings failed: {exc}")
                table = []
            if not table and matches:
                table = self._compute_table(matches)
            return code, {"matches": matches, "standings": table}

        related: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(6, len(entries) or 1)) as pool:
            for code, data in pool.map(lambda e: _fetch(*e), entries):
                related[code] = data
        return related

    def _build_team_contexts(
        self,
        engine: PredictionEngine,
        matches: List[Dict[str, Any]],
        standings: List[Dict[str, Any]],
        related: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Per-team log context: position, league size and points-per-game."""
        contexts: Dict[str, Dict[str, Any]] = {}

        table = self._standings_for(matches, standings)
        size = len(table)
        for row in table:
            played = row.get("played") or 0
            contexts[engine.resolve_name(row["team"])] = {
                "position": row.get("position"),
                "league_size": size,
                "ppg": (row.get("points", 0) / played) if played else None,
                "competition": self._current_code,
            }

        # Teams not in the current log (e.g. foreign CL/EL opponents) get the
        # context of their domestic league log instead. Each distinct feed
        # name is resolved once, not once per match per related league.
        pending: Dict[str, str] = {}
        for m in matches:
            for team in (m.get("home_team"), m.get("away_team")):
                if team and team not in pending:
                    pending[team] = engine.resolve_name(team)
        pending = {t: c for t, c in pending.items() if c not in contexts}

        for code, data in related.items():
            if not pending:
                break
            rel_table = data["standings"]
            if not rel_table:
                continue
            by_name = {r["team"]: r for r in rel_table}
            names = set(by_name)
            for team, canonical in list(pending.items()):
                if canonical in contexts:
                    del pending[team]
                    continue
                resolved = self._resolve_team(team, names, engine)
                if resolved not in by_name:
                    continue
                row = by_name[resolved]
                played = row.get("played") or 0
                contexts[canonical] = {
                    "position": row.get("position"),
                    "league_size": len(rel_table),
                    "ppg": (row.get("points", 0) / played) if played else None,
                    "competition": code,
                }
                del pending[team]
        return contexts

    def refresh(self, refit: bool = False):
        """Public helper to reload all data; `refit` forces a retrain."""
        self._refresh(refit=refit)

    def _refit(self) -> None:
        """Retrain on the loaded data and swap the new engine in. Raises on
        failure, leaving the current engine in place."""
        with self._lock:
            engine = self._train_engine(
                self.matches, self._historical_matches, self._related,
                self._match_source_name,
            )
            contexts = self._build_team_contexts(
                engine, self.matches, self.standings, self._related
            )
            self._engine = engine
            self._team_contexts = contexts

    def refit_model(self):
        """Retrain the prediction model on the currently loaded match data —
        used by startup warm-up so predictions reflect the latest results
        rather than a still-'fresh' model file trained on older data."""
        try:
            self._refit()
        except Exception as exc:
            logger.warning(f"Model refit failed for {self._current_code}: {exc}")

    def refresh_live_matches(self):
        """Lightweight refresh of just the match list — standings and the
        prediction model are untouched. Pairs with utils.bust_live_cache()
        so the fetch bypasses the HTTP TTL and picks up live scores."""
        with self._lock:
            try:
                matches, match_source = self._source.get_matches_with_source(
                    self._current_league
                )
            except Exception as exc:
                logger.warning(f"Live match refresh failed: {exc}")
                return
            if not matches:
                return
            if self._current_league.get("senior_only"):
                matches = _senior_teams_only(matches)
            for m in matches:
                m.setdefault("competition", self._current_code)
            prev_ft = {
                m.get("match_id") for m in self.matches if m.get("status") == "full_time"
            }
            # If games just finished, the model's inputs changed — refit so
            # the very next prediction reflects the latest results.
            new_results = [
                m
                for m in matches
                if m.get("status") == "full_time"
                and m.get("match_id")
                and m.get("match_id") not in prev_ft
            ]
            standings, engine, contexts = self.standings, self._engine, self._team_contexts
            if new_results:
                # Positions/PPG move when results land — patch the table and
                # rebuild contexts against the refitted engine.
                standings = self._apply_results_to_standings(self.standings, new_results)
                try:
                    engine = self._train_engine(
                        matches, self._historical_matches, self._related, match_source
                    )
                except Exception as exc:
                    logger.warning(f"Model refit failed for {self._current_code}: {exc}")
                contexts = self._build_team_contexts(
                    engine, matches, standings, self._related
                )

            self.matches = matches
            self.standings = standings
            self._match_source_name = match_source
            self._engine = engine
            self._team_contexts = contexts
            self._form_index = None
            self._group_match_cache.clear()

    def _apply_results_to_standings(
        self, standings: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return a copy of the standings patched with just-finished results —
        cheaper and safer than recomputing the table, which would lose data
        for leagues whose provider table covers more than the match window.
        Rows are copied so readers of the old table never see it change."""
        if not standings:
            return standings
        standings = [dict(r) for r in standings]
        by_team = {_norm_key(r.get("team", "")): r for r in standings}
        touched = False
        for m in results:
            hs, aws = m.get("home_score"), m.get("away_score")
            h = by_team.get(_norm_key(m.get("home_team", "")))
            a = by_team.get(_norm_key(m.get("away_team", "")))
            if h is None or a is None or hs is None or aws is None:
                continue
            h_pts, a_pts = (3, 0) if hs > aws else ((1, 1) if hs == aws else (0, 3))
            for row, gf, ga, pts in ((h, hs, aws, h_pts), (a, aws, hs, a_pts)):
                row["played"] = (row.get("played") or 0) + 1
                row["won"] = (row.get("won") or 0) + (1 if gf > ga else 0)
                row["drawn"] = (row.get("drawn") or 0) + (1 if gf == ga else 0)
                row["lost"] = (row.get("lost") or 0) + (1 if gf < ga else 0)
                row["goals_for"] = (row.get("goals_for") or 0) + gf
                row["goals_against"] = (row.get("goals_against") or 0) + ga
                row["goal_difference"] = (row.get("goal_difference") or 0) + gf - ga
                row["points"] = (row.get("points") or 0) + pts
            touched = True
        if touched:
            standings.sort(
                key=lambda r: (
                    -(r.get("points") or 0),
                    -(r.get("goal_difference") or 0),
                    -(r.get("goals_for") or 0),
                    r.get("team", ""),
                )
            )
            for i, row in enumerate(standings, 1):
                row["position"] = i
        return standings

    def get_leagues(self) -> List[Tuple[str, str]]:
        # Grouped leagues (e.g. international competitions) are browsed on
        # their dedicated group page, not via the league selector.
        return [
            (code, cfg["name"])
            for code, cfg in self._leagues.items()
            if not cfg.get("group")
        ]

    def get_current_code(self) -> str:
        return self._current_code

    def get_current_name(self) -> str:
        return self._current_league.get("name", self._current_code)

    def get_zones(self) -> Dict[str, int]:
        """Standings highlight zones for the current league, e.g.
        {"qualification": 4, "relegation": 3}. Empty when undefined."""
        return self._current_league.get("zones") or {}

    def get_data_sources(self) -> Dict[str, Optional[str]]:
        return {
            "matches": self._match_source_name,
            "standings": self._standings_source_name,
        }

    def get_teams(self) -> List[str]:
        teams = {r["team"] for r in self.standings}
        for m in self.matches:
            teams.add(m["home_team"])
            teams.add(m["away_team"])
        return sorted(teams)

    def get_matches(self) -> List[Dict[str, Any]]:
        return self.matches

    def get_group_leagues(self, group: str) -> List[Tuple[str, str]]:
        """(code, name) pairs for every league in a group — no fetching."""
        return [
            (code, cfg["name"])
            for code, cfg in self._leagues.items()
            if cfg.get("group") == group
        ]

    def get_group_matches(
        self, group: str, on_progress=None
    ) -> List[Dict[str, Any]]:
        """Return matches from every league tagged with the given group,
        each tagged with its competition code and name. `on_progress` is
        invoked with (code, name) as each league finishes.

        Results are memoised for GROUP_CACHE_TTL so page reruns don't
        re-query every provider. Cached hits still fire `on_progress`
        for each league so the caller's progress UI completes."""
        entries = [
            (code, cfg)
            for code, cfg in self._leagues.items()
            if cfg.get("group") == group
        ]

        entry = self._group_match_cache.get(group)
        if entry and (time.time() - entry[0]) < GROUP_CACHE_TTL:
            if on_progress:
                for code, cfg in entries:
                    try:
                        on_progress(code, cfg.get("name", code))
                    except Exception:
                        pass
            return [dict(m) for m in entry[1]]

        def _fetch(item: Tuple[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
            code, cfg = item
            try:
                league_matches = self._source.get_matches(cfg)
            except Exception as exc:
                logger.warning(f"Group league {code} matches failed: {exc}")
                return []
            if cfg.get("senior_only"):
                league_matches = _senior_teams_only(league_matches)
            for m in league_matches:
                m["competition"] = code
                m["competition_name"] = cfg.get("name", code)
            return league_matches

        matches: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(8, len(entries) or 1)) as pool:
            futures = {pool.submit(_fetch, e): e for e in entries}
            for fut in as_completed(futures):
                code, cfg = futures[fut]
                try:
                    matches.extend(fut.result())
                except Exception as exc:
                    logger.warning(f"Group league {code} matches failed: {exc}")
                if on_progress:
                    try:
                        on_progress(code, cfg.get("name", code))
                    except Exception:
                        pass
        # Safety net: the same fixture can surface under two competitions
        # via different feeds (e.g. a qualifiers feed listing a Nations
        # League game). INTF is the catch-all friendlies bucket, so on a
        # collision the more specific competition's entry wins.
        matches = dedupe_matches(
            matches,
            prefer=lambda new, kept: kept.get("competition") == "INTF"
            and new.get("competition") != "INTF",
        )

        matches.sort(key=lambda x: x["date"] or "")
        self._group_match_cache[group] = (time.time(), matches)
        return [dict(m) for m in matches]

    def get_live_matches(self) -> List[Dict[str, Any]]:
        """Return currently live matches, with a fast path for Football-Data."""
        cfg = self._current_league
        if "football_data" in cfg:
            try:
                live = FootballDataSource().get_live_matches(cfg)
                if live:
                    return live
            except Exception:
                pass
        return [m for m in self.matches if m.get("status") == "live"]

    @staticmethod
    def _compute_table(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return compute_table(matches)

    def _standings_for(
        self, matches: List[Dict[str, Any]], standings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Some competitions (e.g. friendlies) have no meaningful table.
        if self._current_league.get("no_table"):
            return []
        return standings or self._compute_table(matches)

    def get_standings(self) -> List[Dict[str, Any]]:
        return self._standings_for(self.matches, self.standings)

    def _resolve_team(
        self, team: str, pool: Set[str], engine: Optional[PredictionEngine] = None
    ) -> str:
        if team in pool:
            return team
        # Cross-source canonical names (e.g. "Arsenal FC" -> "Arsenal").
        resolved = (engine or self._engine).resolve_name(team)
        if resolved in pool:
            return resolved
        # Case-insensitive exact match as a cheap normalisation step.
        team_lower = team.lower().strip()
        for t in pool:
            if t.lower().strip() == team_lower:
                return t
        matches = difflib.get_close_matches(team, pool, n=1, cutoff=0.75)
        return matches[0] if matches else team

    def get_team_stats(self, team: str) -> Optional[Dict[str, Any]]:
        standings = self.get_standings()
        names = {r["team"] for r in standings}
        resolved = self._resolve_team(team, names)
        for r in standings:
            if r["team"] == resolved:
                return r
        return None

    def _played_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """team -> its completed matches, oldest first. Built once per
        refresh so repeated form lookups don't rescan the whole pool."""
        index = self._form_index
        if index is None:
            matches = self.matches
            pool = list(matches) + list(self._historical_matches)
            for data in self._related.values():
                pool.extend(data["matches"])
            played = [
                m
                for m in dedupe_matches(pool)
                if m.get("status") == "full_time" and m.get("home_score") is not None
            ]
            played.sort(key=lambda m: m.get("date") or "")
            index = {}
            for m in played:
                index.setdefault(m.get("home_team") or "", []).append(m)
                index.setdefault(m.get("away_team") or "", []).append(m)
            index.pop("", None)
            # A writer may have swapped in new matches mid-build — only
            # cache the index if it still describes the current data.
            if self.matches is matches:
                self._form_index = index
        return index

    def get_team_form(self, team: str, n: int = 5) -> List[Dict[str, Any]]:
        """Last-n results for a team, oldest first. Uses current + historical
        + related-competition matches so national teams get real form."""
        played = self._played_index().get(team, [])
        form = []
        for m in played[-n:]:
            if m["home_team"] == team:
                gf, ga = m["home_score"], m["away_score"]
                venue, opponent = "H", m["away_team"]
            else:
                gf, ga = m["away_score"], m["home_score"]
                venue, opponent = "A", m["home_team"]
            if gf > ga:
                result = "W"
            elif gf == ga:
                result = "D"
            else:
                result = "L"
            form.append(
                {
                    "date": m.get("date"),
                    "opponent": opponent,
                    "venue": venue,
                    "result": result,
                    "score": f"{gf}-{ga}",
                }
            )
        return form

    def get_h2h(self, team_a: str, team_b: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """Head-to-head record across current, historical and related-
        competition matches. Names are resolved to the engine's canonical
        form so feed variants (e.g. "Arsenal FC" vs "Arsenal") still pair up."""
        names_a = {team_a, self._engine.resolve_name(team_a)}
        names_b = {team_b, self._engine.resolve_name(team_b)}
        index = self._played_index()
        record = {"played": 0, "a_wins": 0, "b_wins": 0, "draws": 0, "a_gf": 0, "a_ga": 0}
        matches = []
        seen = set()
        for name in names_a:
            for m in index.get(name, []):
                if id(m) in seen:
                    continue
                seen.add(id(m))
                home, away = m["home_team"], m["away_team"]
                if not (
                    (home in names_a and away in names_b)
                    or (home in names_b and away in names_a)
                ):
                    continue
                record["played"] += 1
                a_is_home = home in names_a
                a_gf = m["home_score"] if a_is_home else m["away_score"]
                a_ga = m["away_score"] if a_is_home else m["home_score"]
                record["a_gf"] += a_gf
                record["a_ga"] += a_ga
                if a_gf > a_ga:
                    record["a_wins"] += 1
                elif a_gf == a_ga:
                    record["draws"] += 1
                else:
                    record["b_wins"] += 1
                matches.append(m)
        return record, matches

    def _model_path(self) -> Path:
        return MODEL_CACHE_DIR / f"{self._current_code}_model.json"

    def _model_is_fresh(self) -> bool:
        path = self._model_path()
        if not path.exists():
            return False
        try:
            return (time.time() - path.stat().st_mtime) < MODEL_TTL
        except Exception:
            return False

    def _save_model(self, engine: PredictionEngine) -> None:
        if not engine.fitted:
            return
        try:
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._model_path(), "w", encoding="utf-8") as f:
                json.dump(engine.to_dict(), f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not save model: {exc}")

    def _load_model(self) -> PredictionEngine:
        with open(self._model_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded persisted model for {self._current_code}")
        return PredictionEngine.from_dict(data)

    def _national_seed(self) -> Dict[str, float]:
        """Elo priors for national teams — only used for international leagues."""
        if not self._current_league or self._current_league.get("group") != "international":
            return {}
        ranks = _load_national_rankings()
        if not ranks:
            return {}
        seed = dict(ranks)
        for feed_name, rank_name in _NATIONAL_ALIASES.items():
            if rank_name in ranks:
                seed[feed_name] = ranks[rank_name]
        return seed

    def _team_value_seed(self) -> Dict[str, float]:
        """Elo priors from squad market values — only for club leagues.
        Log-scaled so the ~100x spread between a super-club and a minnow
        maps to a few hundred Elo points around the pool mean."""
        if not self._current_league or self._current_league.get("group") == "international":
            return {}
        values = _load_team_values()
        if not values:
            return {}
        ref = math.exp(sum(math.log(v) for v in values.values()) / len(values))
        return {
            name: 1500.0 + VALUE_ELO_PER_LN * (math.log(v) - math.log(ref))
            for name, v in values.items()
        }

    def _elo_seed(self) -> Dict[str, float]:
        """Combined starting-Elo priors; national rankings win where both
        apply (club values are meaningless for international pools)."""
        return {**self._team_value_seed(), **self._national_seed()}

    def _seed_elo(self, engine: PredictionEngine) -> None:
        """Merge ranking/value priors into the engine's Elo table. Trained
        values win where they exist; seeded ratings make the fallback path
        useful for teams with no completed-match history."""
        seed = self._elo_seed()
        if seed:
            engine.elo = {**seed, **engine.elo}
            engine.elo_prior.update(
                {
                    _TEAM_KEY_ALIASES.get(_norm_key(k), _norm_key(k)): float(v)
                    for k, v in seed.items()
                }
            )

    def _train_engine(
        self,
        matches: List[Dict[str, Any]],
        historical: List[Dict[str, Any]],
        related: Dict[str, Dict[str, Any]],
        match_source: Optional[str],
    ) -> PredictionEngine:
        """Train, persist and seed a new engine on current + historical
        matches. Never touches self — callers swap the result in."""
        training = list(matches)
        if match_source != "openfootball" and historical:
            training.extend(historical)
        # Other tournaments (e.g. domestic leagues of CL/EL clubs).
        for data in related.values():
            training.extend(data["matches"])
        engine = PredictionEngine()
        engine.fit(
            training,
            initial_elo=self._elo_seed(),
            primary_competition=self._current_code,
        )
        self._save_model(engine)
        self._seed_elo(engine)
        return engine

    def predict(self, home_team: str, away_team: str, neutral: bool = False) -> Dict[str, Any]:
        """Return a prediction from the currently trained model."""
        # One read of each so a concurrent swap can't pair a new engine
        # with old contexts mid-prediction.
        engine, team_contexts = self._engine, self._team_contexts
        home = self._resolve_team(home_team, engine.teams, engine)
        away = self._resolve_team(away_team, engine.teams, engine)
        contexts = {
            home: team_contexts.get(home),
            away: team_contexts.get(away),
        }
        return engine.predict(
            home, away, neutral, competition=self._current_code, contexts=contexts
        )

    def predict_fixture(self, match: Dict[str, Any], neutral: bool = False) -> Dict[str, Any]:
        """Predict a fixture row and log it so its accuracy is scored once
        the result lands."""
        pred = self.predict(match["home_team"], match["away_team"], neutral=neutral)
        prediction_log.record(self._current_code, [(match, pred)])
        return pred

    def track_predictions(self, horizon_days: int = 7) -> None:
        """Score logged predictions whose results are in, then log fresh
        predictions for every fixture kicking off within `horizon_days` —
        so accuracy covers the whole slate, not just what users clicked."""
        matches = self.matches
        prediction_log.settle(self._current_code, matches)
        now = sast_now()
        limit = now + timedelta(days=horizon_days)
        pairs = []
        for m in matches:
            if m.get("status") != "scheduled" or not m.get("match_id"):
                continue
            kickoff = prediction_log.kickoff_sast(m.get("date"))
            if not kickoff or not (now < kickoff <= limit):
                continue
            try:
                pairs.append((m, self.predict(m["home_team"], m["away_team"])))
            except Exception as exc:
                logger.warning(f"Tracking prediction failed for {m.get('match_id')}: {exc}")
        prediction_log.record(self._current_code, pairs)

    def _training_size(self) -> int:
        related = sum(len(d["matches"]) for d in self._related.values())
        return len(self.matches) + len(self._historical_matches) + related

    def train(self) -> Dict[str, Any]:
        """Force a fresh retrain and return model status."""
        try:
            self._refit()
        except Exception as exc:
            logger.warning(f"Model retraining failed: {exc}")
            return {
                "last_trained": None,
                "matches_used": self._training_size(),
                "teams_in_model": 0,
                "fitted": False,
                "error": str(exc),
            }
        return {
            "last_trained": self._engine.last_trained,
            "matches_used": self._training_size(),
            "teams_in_model": len(self._engine.teams),
            "fitted": self._engine.fitted,
        }
