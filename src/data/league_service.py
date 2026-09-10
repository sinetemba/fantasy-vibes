"""High-level service that loads league data and makes predictions."""

import difflib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from .prediction_engine import PredictionEngine
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


class LeagueService:
    """Loads fixtures, tables and provides match predictions for a league."""

    def __init__(self, league_code: str = "PL"):
        self._leagues = self._load_leagues()
        self._match_sources = [
            FixtureDownloadSource(),
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
        self.matches = self._source.get_matches(self._current_league)
        self._match_source_name = self._detect_source(self._match_sources, "matches")
        self.standings = self._source.get_standings(self._current_league)
        self._standings_source_name = self._detect_source(self._standings_sources, "standings")
        # Fallback to a live table computed from results if no source provides standings.
        if not self.standings and self.matches:
            self._standings_source_name = "computed"

        # Pull openfootball history as extra training data when the live source is different
        self._historical_matches = []
        if "openfootball" in self._current_league and self._match_source_name != "openfootball":
            try:
                self._historical_matches = OpenfootballSource().get_matches(self._current_league)
            except Exception:
                self._historical_matches = []

        # Pull football-data historical seasons as extra training data
        if "football_data" in self._current_league:
            for hist_season in self._current_league["football_data"].get("historical", []):
                try:
                    self._historical_matches.extend(
                        FootballDataSource().get_matches(self._current_league, season=hist_season)
                    )
                except Exception:
                    pass

        # Pull TheSportsDB historical seasons as extra training data
        if "thesportsdb" in self._current_league:
            for hist_season in self._current_league["thesportsdb"].get("historical", []):
                try:
                    self._historical_matches.extend(
                        TheSportsDBSource().get_matches(self._current_league, season=hist_season)
                    )
                except Exception:
                    pass

        # Pull FixtureDownload historical seasons as extra training data
        if "fixturedownload" in self._current_league:
            for hist_season in self._current_league["fixturedownload"].get("historical", []):
                try:
                    self._historical_matches.extend(
                        FixtureDownloadSource().get_matches(self._current_league, season=hist_season)
                    )
                except Exception:
                    pass

        self._load_or_fit_model()

    def _detect_source(self, sources: List[Any], method: str) -> Optional[str]:
        for s in sources:
            if not s.is_available(self._current_league):
                continue
            fn = getattr(s, f"get_{method}")
            data = fn(self._current_league)
            if data:
                return s.name
        return None

    def get_leagues(self) -> List[Tuple[str, str]]:
        return [(code, cfg["name"]) for code, cfg in self._leagues.items()]

    def get_current_code(self) -> str:
        return self._current_code

    def get_current_name(self) -> str:
        return self._current_league.get("name", self._current_code)

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

    def _compute_table_from_matches(self) -> List[Dict[str, Any]]:
        table: Dict[str, Dict[str, Any]] = {}
        all_teams = set()

        for m in self.matches:
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

    def get_standings(self) -> List[Dict[str, Any]]:
        if not self.standings:
            return self._compute_table_from_matches()
        return self.standings

    def _resolve_team(self, team: str, pool: Set[str]) -> str:
        if team in pool:
            return team
        matches = difflib.get_close_matches(team, pool, n=1, cutoff=0.6)
        return matches[0] if matches else team

    def get_team_stats(self, team: str) -> Optional[Dict[str, Any]]:
        standings = self.get_standings()
        names = {r["team"] for r in standings}
        resolved = self._resolve_team(team, names)
        for r in standings:
            if r["team"] == resolved:
                return r
        return None

    def get_team_form(self, team: str, n: int = 5) -> List[Dict[str, Any]]:
        form = []
        for m in reversed(self.matches):
            if m["status"] != "full_time" or m["home_score"] is None:
                continue
            if m["home_team"] == team:
                gf, ga = m["home_score"], m["away_score"]
                venue = "H"
                opponent = m["away_team"]
            elif m["away_team"] == team:
                gf, ga = m["away_score"], m["home_score"]
                venue = "A"
                opponent = m["home_team"]
            else:
                continue
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
            if len(form) >= n:
                break
        return form[::-1]

    def get_h2h(self, team_a: str, team_b: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        record = {"played": 0, "a_wins": 0, "b_wins": 0, "draws": 0, "a_gf": 0, "a_ga": 0}
        matches = []
        for m in self.matches:
            if m["status"] != "full_time" or m["home_score"] is None:
                continue
            if {m["home_team"], m["away_team"]} != {team_a, team_b}:
                continue
            record["played"] += 1
            a_is_home = m["home_team"] == team_a
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

    def _load_or_fit_model(self) -> None:
        if self._model_is_fresh():
            try:
                self._load_model()
                return
            except Exception as exc:
                logger.warning(f"Persisted model invalid, retraining: {exc}")
        self._fit_model()

    def _fit_model(self) -> None:
        """Train the prediction engine on current + historical matches."""
        training = list(self.matches)
        if self._match_source_name != "openfootball" and self._historical_matches:
            training.extend(self._historical_matches)
        self._engine = PredictionEngine()
        self._engine.fit(training)
        self._save_model()

    def predict(self, home_team: str, away_team: str, neutral: bool = False) -> Dict[str, Any]:
        """Return a prediction from the currently trained model."""
        home = self._resolve_team(home_team, self._engine.teams)
        away = self._resolve_team(away_team, self._engine.teams)
        return self._engine.predict(home, away, neutral)

    def train(self) -> Dict[str, Any]:
        """Force a fresh retrain and return model status."""
        self._fit_model()
        return {
            "last_trained": self._engine.last_trained,
            "matches_used": len(self.matches) + len(self._historical_matches),
            "teams_in_model": len(self._engine.teams),
            "fitted": self._engine.fitted,
        }
