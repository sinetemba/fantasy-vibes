"""Lightweight trainable prediction engine for league matches."""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def _expected_score(elo_a: float, elo_b: float, a_field: float = 0.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a - a_field) / 400.0))


def _draw_prob(elo_diff: float) -> float:
    return max(0.15, 0.35 - abs(elo_diff) / 2000.0)


class PredictionEngine:
    """Fits attack/defense, Elo and form from match history then predicts."""

    def __init__(self):
        self.fitted = False
        self.last_trained: Optional[str] = None

        self.league_avg_home: float = 1.4
        self.league_avg_away: float = 1.1
        self.home_advantage: float = 1.15

        self.home_attack: Dict[str, float] = {}
        self.home_defense: Dict[str, float] = {}
        self.away_attack: Dict[str, float] = {}
        self.away_defense: Dict[str, float] = {}
        self.elo: Dict[str, float] = {}
        self.form_ppg: Dict[str, float] = {}
        self.teams: set = set()

    def fit(self, matches: List[Dict[str, Any]]) -> None:
        """Train on a list of full-time match dicts."""
        records = [
            m
            for m in matches
            if m.get("status") == "full_time" and m.get("home_score") is not None
        ]
        if not records:
            self.fitted = False
            return

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["home_score", "away_score"])

        self.teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

        self.league_avg_home = max(df["home_score"].mean(), 0.1)
        self.league_avg_away = max(df["away_score"].mean(), 0.1)
        self.home_advantage = max(1.05, min(1.4, self.league_avg_home / self.league_avg_away))

        # Attack/defense ratings
        home_goals = df.groupby("home_team")["home_score"].mean().to_dict()
        home_conceded = df.groupby("home_team")["away_score"].mean().to_dict()
        away_goals = df.groupby("away_team")["away_score"].mean().to_dict()
        away_conceded = df.groupby("away_team")["home_score"].mean().to_dict()

        self.home_attack = {t: g / self.league_avg_home for t, g in home_goals.items()}
        self.home_defense = {t: g / self.league_avg_away for t, g in home_conceded.items()}
        self.away_attack = {t: g / self.league_avg_away for t, g in away_goals.items()}
        self.away_defense = {t: g / self.league_avg_home for t, g in away_conceded.items()}

        # Elo ratings
        self._fit_elo(df)

        # Recent form (last 5 matches)
        self._fit_form(df)

        self.fitted = True
        self.last_trained = datetime.now().isoformat()
        logger.info(f"PredictionEngine trained on {len(df)} matches")

    def _fit_elo(self, df: pd.DataFrame, k: float = 30.0, home_field: float = 70.0):
        self.elo = {t: 1500.0 for t in self.teams}
        for _, row in df.sort_values("date").iterrows():
            home = row["home_team"]
            away = row["away_team"]
            hs, aws = row["home_score"], row["away_score"]

            expected = _expected_score(self.elo[home] + home_field, self.elo[away])
            if hs > aws:
                actual = 1.0
            elif hs == aws:
                actual = 0.5
            else:
                actual = 0.0

            delta = k * (actual - expected)
            self.elo[home] += delta
            self.elo[away] -= delta

        # Ensure every team has a default rating
        for t in self.teams:
            self.elo.setdefault(t, 1500.0)

    def _fit_form(self, df: pd.DataFrame, n: int = 5):
        form = {t: [] for t in self.teams}
        for _, row in df.sort_values("date").iterrows():
            home, away = row["home_team"], row["away_team"]
            hs, aws = row["home_score"], row["away_score"]
            if hs > aws:
                home_pts, away_pts = 3, 0
            elif hs == aws:
                home_pts, away_pts = 1, 1
            else:
                home_pts, away_pts = 0, 3
            form[home].append(home_pts)
            form[away].append(away_pts)

        self.form_ppg = {}
        for t, pts in form.items():
            recent = pts[-n:]
            max_pts = len(recent) * 3
            self.form_ppg[t] = (sum(recent) / max_pts) if max_pts else 0.5

    def _form_factor(self, team: str) -> float:
        ppg = self.form_ppg.get(team, 0.5)
        # Scale: +10% if perfect form, -10% if zero form
        return 1.0 + 0.2 * (ppg - 0.5)

    def predict(self, home_team: str, away_team: str, neutral: bool = False) -> Dict[str, Any]:
        if not self.fitted or home_team not in self.teams or away_team not in self.teams:
            return self._fallback(home_team, away_team)

        home_attack = self.home_attack.get(home_team, 1.0) * self._form_factor(home_team)
        home_defense = self.home_defense.get(home_team, 1.0) / self._form_factor(home_team)
        away_attack = self.away_attack.get(away_team, 1.0) * self._form_factor(away_team)
        away_defense = self.away_defense.get(away_team, 1.0) / self._form_factor(away_team)

        ha = 1.0 if neutral else self.home_advantage
        ah = 1.0 / ha

        home_exp = max(0.1, self.league_avg_home * home_attack * away_defense * ha)
        away_exp = max(0.1, self.league_avg_away * away_attack * home_defense * ah)

        # Elo-based outcome probabilities
        field = 0.0 if neutral else 70.0
        elo_diff = self.elo[home_team] - self.elo[away_team] + field
        elo_win = _expected_score(self.elo[home_team], self.elo[away_team], field)
        draw = _draw_prob(elo_diff)
        remaining = 1.0 - draw
        elo_home_win = elo_win * remaining
        elo_away_win = (1.0 - elo_win) * remaining

        # Poisson-based outcome and score matrix
        max_goals = 6
        matrix = [[0.0 for _ in range(max_goals + 1)] for _ in range(max_goals + 1)]
        poisson_home_win = poisson_draw = poisson_away_win = 0.0
        over = under = btts_yes = 0.0
        best_score = "0-0"
        max_prob = 0.0

        for i in range(max_goals + 1):
            p_i = _poisson_pmf(i, home_exp)
            for j in range(max_goals + 1):
                p_j = _poisson_pmf(j, away_exp)
                p = p_i * p_j
                matrix[i][j] = p

                if i > j:
                    poisson_home_win += p
                elif i == j:
                    poisson_draw += p
                else:
                    poisson_away_win += p

                if i + j > 2.5:
                    over += p
                else:
                    under += p

                if i > 0 and j > 0:
                    btts_yes += p

                if p > max_prob:
                    max_prob = p
                    best_score = f"{i}-{j}"

        # Blend Elo and Poisson win/draw/loss
        home_win = 0.5 * poisson_home_win + 0.5 * elo_home_win
        draw = 0.5 * poisson_draw + 0.5 * draw
        away_win = 0.5 * poisson_away_win + 0.5 * elo_away_win
        total = home_win + draw + away_win
        if total:
            home_win /= total
            draw /= total
            away_win /= total

        return {
            "home_team": home_team,
            "away_team": away_team,
            "model": "trained_poisson_elo",
            "expected_goals": {"home": round(home_exp, 2), "away": round(away_exp, 2)},
            "outcome_probabilities": {
                "home_win": round(home_win, 4),
                "draw": round(draw, 4),
                "away_win": round(away_win, 4),
            },
            "predicted_score": {
                "score": best_score,
                "probability": round(max_prob, 4),
            },
            "over_under_2_5": {"over": round(over, 4), "under": round(under, 4)},
            "btts": {"yes": round(btts_yes, 4), "no": round(1.0 - btts_yes, 4)},
            "team_attack_params": {
                "home": round(self.home_attack.get(home_team, 1.0), 3),
                "away": round(self.away_attack.get(away_team, 1.0), 3),
            },
            "team_defense_params": {
                "home": round(self.home_defense.get(home_team, 1.0), 3),
                "away": round(self.away_defense.get(away_team, 1.0), 3),
            },
            "score_matrix": np.array(matrix),
            "last_trained": self.last_trained,
        }

    def _fallback(self, home_team: str, away_team: str) -> Dict[str, Any]:
        home_elo = self.elo.get(home_team, 1500)
        away_elo = self.elo.get(away_team, 1500)
        expected_home = _expected_score(home_elo, away_elo)
        draw = 0.25
        remaining = 1.0 - draw
        home_win = expected_home * remaining
        away_win = (1.0 - expected_home) * remaining
        home_exp = max(0.5, 2.5 * (home_elo / (home_elo + away_elo)) * 1.1)
        away_exp = max(0.5, 2.5 * (away_elo / (home_elo + away_elo)))

        return {
            "home_team": home_team,
            "away_team": away_team,
            "model": "elo_fallback",
            "expected_goals": {"home": round(home_exp, 2), "away": round(away_exp, 2)},
            "outcome_probabilities": {
                "home_win": round(home_win, 4),
                "draw": round(draw, 4),
                "away_win": round(away_win, 4),
            },
            "predicted_score": {
                "score": f"{round(home_exp)}-{round(away_exp)}",
                "probability": 0.0,
            },
            "over_under_2_5": {
                "over": round(0.5 + (home_exp + away_exp - 2.5) * 0.1, 4),
                "under": round(0.5 - (home_exp + away_exp - 2.5) * 0.1, 4),
            },
            "btts": {
                "yes": round(1.0 - math.exp(-home_exp) - math.exp(-away_exp) + math.exp(-home_exp - away_exp), 4),
                "no": 0.0,
            },
            "team_attack_params": {"home": 1.0, "away": 1.0},
            "team_defense_params": {"home": 1.0, "away": 1.0},
            "score_matrix": np.zeros((7, 7)),
            "last_trained": self.last_trained,
        }
