"""High-level service that loads league data and makes predictions."""

import difflib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from .prediction_engine import PredictionEngine
from .utils import compute_table
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

    def _refresh(self):
        self._group_match_cache.clear()
        self._form_index = None

        # Matches and standings use disjoint provider lists, so they fetch
        # in parallel. Historical pulls depend on which source won the
        # match cascade, so they start once matches land and overlap the
        # standings fetch.
        with ThreadPoolExecutor(max_workers=3) as pool:
            matches_fut = pool.submit(self._source.get_matches, self._current_league)
            standings_fut = pool.submit(self._source.get_standings, self._current_league)
            self.matches = matches_fut.result()
            self._match_source_name = self._source.last_matches_source
            hist_fut = pool.submit(self._fetch_historical)
            self.standings = standings_fut.result()
            self._standings_source_name = self._source.last_standings_source
            self._historical_matches = hist_fut.result()

        if self._current_league.get("senior_only"):
            self.matches = _senior_teams_only(self.matches)
        # Fallback to a live table computed from results if no source provides standings.
        if not self.standings and self.matches and not self._current_league.get("no_table"):
            self._standings_source_name = "computed"

        # Tag every match with its competition so the engine can track
        # per-tournament form (e.g. domestic league vs Champions League).
        for m in self.matches:
            m.setdefault("competition", self._current_code)
        for m in self._historical_matches:
            m.setdefault("competition", self._current_code)

        # Pull related competitions (e.g. the domestic leagues of CL/EL clubs)
        # so predictions can use each team's own log position and other-
        # tournament form.
        self._related = self._load_related()

        self._load_or_fit_model()
        self._team_contexts = self._build_team_contexts()

    def _fetch_historical(self) -> List[Dict[str, Any]]:
        """Pull configured historical/archived seasons as extra training data."""
        historical: List[Dict[str, Any]] = []

        # Pull openfootball history as extra training data when the live source is different
        if "openfootball" in self._current_league and self._match_source_name != "openfootball":
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

    def _build_team_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Per-team log context: position, league size and points-per-game."""
        contexts: Dict[str, Dict[str, Any]] = {}

        standings = self.get_standings()
        size = len(standings)
        for row in standings:
            played = row.get("played") or 0
            contexts[self._engine.resolve_name(row["team"])] = {
                "position": row.get("position"),
                "league_size": size,
                "ppg": (row.get("points", 0) / played) if played else None,
                "competition": self._current_code,
            }

        # Teams not in the current log (e.g. foreign CL/EL opponents) get the
        # context of their domestic league log instead.
        for code, data in self._related.items():
            table = data["standings"]
            if not table:
                continue
            names = {r["team"] for r in table}
            by_name = {r["team"]: r for r in table}
            for m in self.matches:
                for team in (m.get("home_team"), m.get("away_team")):
                    if not team:
                        continue
                    canonical = self._engine.resolve_name(team)
                    if canonical in contexts:
                        continue
                    resolved = self._resolve_team(team, names)
                    if resolved not in by_name:
                        continue
                    row = by_name[resolved]
                    played = row.get("played") or 0
                    contexts[canonical] = {
                        "position": row.get("position"),
                        "league_size": len(table),
                        "ppg": (row.get("points", 0) / played) if played else None,
                        "competition": code,
                    }
        return contexts

    def refresh(self):
        """Public helper to reload data and retrain the model."""
        self._refresh()

    def refresh_live_matches(self):
        """Lightweight refresh of just the match list — standings and the
        prediction model are untouched. Pairs with utils.bust_live_cache()
        so the fetch bypasses the HTTP TTL and picks up live scores."""
        try:
            matches = self._source.get_matches(self._current_league)
        except Exception as exc:
            logger.warning(f"Live match refresh failed: {exc}")
            return
        if not matches:
            return
        if self._current_league.get("senior_only"):
            matches = _senior_teams_only(matches)
        for m in matches:
            m.setdefault("competition", self._current_code)
        self.matches = matches
        self._match_source_name = self._source.last_matches_source
        self._form_index = None
        self._group_match_cache.clear()

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
        seen: Dict[Tuple[str, str, str], int] = {}
        unique: List[Dict[str, Any]] = []
        for m in matches:
            key = (
                (m.get("home_team") or "").strip().lower(),
                (m.get("away_team") or "").strip().lower(),
                (m.get("date") or "")[:10],
            )
            prev = seen.get(key)
            if prev is None:
                seen[key] = len(unique)
                unique.append(m)
            elif unique[prev].get("competition") == "INTF" and m.get("competition") != "INTF":
                unique[prev] = m
        matches = unique

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

    def _compute_table_from_matches(self) -> List[Dict[str, Any]]:
        return self._compute_table(self.matches)

    def get_standings(self) -> List[Dict[str, Any]]:
        # Some competitions (e.g. friendlies) have no meaningful table.
        if self._current_league.get("no_table"):
            return []
        if not self.standings:
            return self._compute_table_from_matches()
        return self.standings

    def _resolve_team(self, team: str, pool: Set[str]) -> str:
        if team in pool:
            return team
        # Cross-source canonical names (e.g. "Arsenal FC" -> "Arsenal").
        resolved = self._engine.resolve_name(team)
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
        if self._form_index is None:
            pool = list(self.matches) + list(self._historical_matches)
            for data in self._related.values():
                pool.extend(data["matches"])
            played = [
                m
                for m in pool
                if m.get("status") == "full_time" and m.get("home_score") is not None
            ]
            played.sort(key=lambda m: m.get("date") or "")
            index: Dict[str, List[Dict[str, Any]]] = {}
            for m in played:
                index.setdefault(m.get("home_team") or "", []).append(m)
                index.setdefault(m.get("away_team") or "", []).append(m)
            index.pop("", None)
            self._form_index = index
        return self._form_index

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

    def _save_model(self) -> None:
        if not self._engine.fitted:
            return
        try:
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._model_path(), "w", encoding="utf-8") as f:
                json.dump(self._engine.to_dict(), f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not save model: {exc}")

    def _load_model(self) -> None:
        with open(self._model_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        self._engine = PredictionEngine.from_dict(data)
        logger.info(f"Loaded persisted model for {self._current_code}")

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

    def _load_or_fit_model(self) -> None:
        if self._model_is_fresh():
            try:
                self._load_model()
            except Exception as exc:
                logger.warning(f"Persisted model invalid, retraining: {exc}")
                self._fit_model()
        else:
            self._fit_model()
        self._seed_national_elo()

    def _seed_national_elo(self) -> None:
        """Merge ranking priors into the engine's Elo table. Trained values
        win where they exist; seeded ratings make the fallback path useful
        for national teams with no completed-match history."""
        seed = self._national_seed()
        if seed:
            self._engine.elo = {**seed, **self._engine.elo}

    def _fit_model(self) -> None:
        """Train the prediction engine on current + historical matches."""
        training = list(self.matches)
        if self._match_source_name != "openfootball" and self._historical_matches:
            training.extend(self._historical_matches)
        # Other tournaments (e.g. domestic leagues of CL/EL clubs).
        for data in self._related.values():
            training.extend(data["matches"])
        self._engine = PredictionEngine()
        self._engine.fit(training, initial_elo=self._national_seed())
        self._save_model()

    def predict(self, home_team: str, away_team: str, neutral: bool = False) -> Dict[str, Any]:
        """Return a prediction from the currently trained model."""
        home = self._resolve_team(home_team, self._engine.teams)
        away = self._resolve_team(away_team, self._engine.teams)
        contexts = {
            home: self._team_contexts.get(home),
            away: self._team_contexts.get(away),
        }
        return self._engine.predict(
            home, away, neutral, competition=self._current_code, contexts=contexts
        )

    def _training_size(self) -> int:
        related = sum(len(d["matches"]) for d in self._related.values())
        return len(self.matches) + len(self._historical_matches) + related

    def train(self) -> Dict[str, Any]:
        """Force a fresh retrain and return model status."""
        try:
            self._fit_model()
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
