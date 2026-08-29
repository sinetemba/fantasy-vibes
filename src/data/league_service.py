"""High-level service that loads league data and makes predictions."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .prediction_engine import PredictionEngine
from .sources.composite import CompositeDataSource
from .sources.football_data import FootballDataSource
from .sources.openfootball import OpenfootballSource
from .sources.thesportsdb import TheSportsDBSource

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEAGUES_PATH = PROJECT_ROOT / "data" / "leagues.json"


class LeagueService:
    """Loads fixtures, tables and provides match predictions for a league."""

    def __init__(self, league_code: str = "PL"):
        self._leagues = self._load_leagues()
        self._match_sources = [
            FootballDataSource(),
            OpenfootballSource(),
            TheSportsDBSource(),
        ]
        self._standings_sources = [
            FootballDataSource(),
            OpenfootballSource(),
            TheSportsDBSource(),
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

        # Pull openfootball history as extra training data when the live source is different
        self._historical_matches = []
        if "openfootball" in self._current_league and self._match_source_name != "openfootball":
            try:
                self._historical_matches = OpenfootballSource().get_matches(self._current_league)
            except Exception:
                self._historical_matches = []

        self._fit_model()

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
        return sorted({r["team"] for r in self.standings})

    def get_matches(self) -> List[Dict[str, Any]]:
        return self.matches

    def get_standings(self) -> List[Dict[str, Any]]:
        return self.standings

    def get_team_stats(self, team: str) -> Optional[Dict[str, Any]]:
        for r in self.standings:
            if r["team"] == team:
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

    def _fit_model(self) -> None:
        """Train the prediction engine on current + historical matches."""
        training = list(self.matches)
        if self._match_source_name != "openfootball" and self._historical_matches:
            training.extend(self._historical_matches)
        self._engine = PredictionEngine()
        self._engine.fit(training)

    def predict(self, home_team: str, away_team: str, neutral: bool = False) -> Dict[str, Any]:
        """Return a prediction from the currently trained model."""
        return self._engine.predict(home_team, away_team, neutral)

    def train(self) -> Dict[str, Any]:
        """Force a fresh retrain and return model status."""
        self._fit_model()
        return {
            "last_trained": self._engine.last_trained,
            "matches_used": len(self.matches) + len(self._historical_matches),
            "teams_in_model": len(self._engine.teams),
            "fitted": self._engine.fitted,
        }
