"""Lightweight trainable prediction engine for league matches."""

import logging
import math
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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


RECENCY_HALF_LIFE_DAYS = 180.0

# Shrink per-team attack/defense ratings toward the league mean when a team
# has few matches — otherwise small samples produce unrealistic xG.
ATTACK_SHRINK_GAMES = 8.0
MAX_EXPECTED_GOALS = 5.0

# Tuning weights for the context (log position / points-per-game) adjustment.
POSITION_WEIGHT = 0.24
PPG_WEIGHT = 0.20
CONTEXT_FACTOR_MIN = 0.85
CONTEXT_FACTOR_MAX = 1.20

# Tokens that carry no identity when comparing club names across sources,
# e.g. "Arsenal FC" vs "Arsenal" or "Real Madrid CF" vs "Real Madrid".
TEAM_STOP_TOKENS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ssc", "fk", "sk", "bk", "if",
    "bv", "sv", "cd", "ud", "rc", "rcd", "ogc", "tsg", "vfl", "vfb", "fsv",
    "club", "the", "de",
}


def _norm_key(name: str) -> str:
    """Normalise a team name for cross-source / cross-competition matching."""
    tokens = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    tokens = [t for t in tokens if t not in TEAM_STOP_TOKENS and not t.isdigit()]
    return " ".join(tokens) or name.lower().strip()


def _weighted_avg(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


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
        self.comp_form_ppg: Dict[str, float] = {}
        self.name_index: Dict[str, str] = {}
        self.teams: set = set()

    def fit(
        self,
        matches: List[Dict[str, Any]],
        initial_elo: Optional[Dict[str, float]] = None,
    ) -> None:
        """Train on a list of full-time match dicts. `initial_elo` seeds
        starting ratings (e.g. national-team rankings) instead of 1500."""
        self._initial_elo = initial_elo or {}
        records = [
            m
            for m in matches
            if m.get("status") == "full_time" and m.get("home_score") is not None
        ]
        if not records:
            self.fitted = False
            return

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["home_score", "away_score", "date"])
        if "competition" not in df.columns:
            df["competition"] = "default"
        df["competition"] = df["competition"].fillna("default")

        # Canonicalise team names so the same club is merged across sources
        # and competitions (e.g. "Arsenal FC" in one feed, "Arsenal" in another).
        name_map = self._canonical_map(
            df["home_team"].tolist() + df["away_team"].tolist()
        )
        df["home_team"] = df["home_team"].map(name_map)
        df["away_team"] = df["away_team"].map(name_map)

        self.teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

        # Recency weighting: more recent matches count more.
        latest = df["date"].max()
        df["days_ago"] = (latest - df["date"]).dt.days.astype(float)
        df["weight"] = np.exp(-np.log(2) / RECENCY_HALF_LIFE_DAYS * df["days_ago"])
        df["weight"] = df["weight"].clip(lower=0.01)

        self.league_avg_home = max(_weighted_avg(df["home_score"].values, df["weight"].values), 0.1)
        self.league_avg_away = max(_weighted_avg(df["away_score"].values, df["weight"].values), 0.1)
        self.home_advantage = max(1.05, min(1.4, self.league_avg_home / self.league_avg_away))

        # Attack/defense ratings with recency weighting, shrunk toward the
        # league mean for small samples (a team that played 3 games shouldn't
        # get a 4x attack rating).
        def _shrunk_ratios(group_col: str, score_col: str, avg: float) -> Dict[str, float]:
            ratios: Dict[str, float] = {}
            for team, grp in df.groupby(group_col):
                w = grp["weight"].values
                g = grp[score_col].values
                wsum = float(w.sum())
                raw = _weighted_avg(g, w)
                shrunk = (raw * wsum + ATTACK_SHRINK_GAMES * avg) / (
                    wsum + ATTACK_SHRINK_GAMES
                )
                ratios[team] = shrunk / avg
            return ratios

        self.home_attack = _shrunk_ratios("home_team", "home_score", self.league_avg_home)
        self.home_defense = _shrunk_ratios("home_team", "away_score", self.league_avg_away)
        self.away_attack = _shrunk_ratios("away_team", "away_score", self.league_avg_away)
        self.away_defense = _shrunk_ratios("away_team", "home_score", self.league_avg_home)

        # Elo ratings
        self._fit_elo(df)

        # Recent form (weighted last 10 matches)
        self._fit_form(df, n=10)

        self.fitted = True
        self.last_trained = datetime.now().isoformat()
        logger.info(f"PredictionEngine trained on {len(df)} matches")

    def _canonical_map(self, names: List[str]) -> Dict[str, str]:
        """Map raw team names to canonical names shared across sources."""
        counts = Counter(names)
        key_to_names: Dict[str, List[str]] = {}
        for name in counts:
            key_to_names.setdefault(_norm_key(name), []).append(name)

        # Merge keys where one is a token-prefix of the other
        # (e.g. "tottenham" and "tottenham hotspur").
        keys = sorted(key_to_names, key=len)
        merged: Dict[str, str] = {}
        for key in keys:
            target = key
            for shorter in keys:
                if len(shorter) >= len(key):
                    break
                if key.startswith(shorter + " "):
                    target = merged.get(shorter, shorter)
                    break
            merged[key] = target

        # Group raw names by their merged key and pick one canonical name per
        # group (the most frequent raw variant).
        groups: Dict[str, List[str]] = {}
        for key, canonical_key in merged.items():
            groups.setdefault(canonical_key, []).extend(key_to_names[key])

        name_map: Dict[str, str] = {}
        self.name_index = {}
        for canonical_key, raws in groups.items():
            canonical = max(raws, key=lambda n: counts[n])
            for raw in raws:
                name_map[raw] = canonical
                self.name_index[raw] = canonical
            self.name_index[canonical_key] = canonical
        return name_map

    def resolve_name(self, name: str) -> str:
        """Resolve a raw team name to the canonical model name."""
        if name in self.teams:
            return name
        if name in self.name_index:
            return self.name_index[name]
        key = _norm_key(name)
        if key in self.name_index:
            return self.name_index[key]
        return name

    def _fit_elo(self, df: pd.DataFrame, k: float = 30.0, home_field: float = 70.0):
        priors = getattr(self, "_initial_elo", {})
        self.elo = {t: float(priors.get(t, 1500.0)) for t in self.teams}
        for _, row in df.sort_values("date").iterrows():
            home = row["home_team"]
            away = row["away_team"]
            hs, aws = row["home_score"], row["away_score"]
            weight = row.get("weight", 1.0)

            expected = _expected_score(self.elo[home] + home_field, self.elo[away])
            if hs > aws:
                actual = 1.0
            elif hs == aws:
                actual = 0.5
            else:
                actual = 0.0

            delta = k * weight * (actual - expected)
            self.elo[home] += delta
            self.elo[away] -= delta

        # Ensure every team has a default rating
        for t in self.teams:
            self.elo.setdefault(t, 1500.0)

    def _fit_form(self, df: pd.DataFrame, n: int = 10, comp_n: int = 6):
        form: Dict[str, List[Dict[str, Any]]] = {t: [] for t in self.teams}
        comp_form: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for _, row in df.sort_values("date").iterrows():
            home, away = row["home_team"], row["away_team"]
            comp = row.get("competition") or "default"
            hs, aws = row["home_score"], row["away_score"]
            if hs > aws:
                home_pts, away_pts = 3, 0
            elif hs == aws:
                home_pts, away_pts = 1, 1
            else:
                home_pts, away_pts = 0, 3
            weight = row.get("weight", 1.0)
            form[home].append({"pts": home_pts, "weight": weight})
            form[away].append({"pts": away_pts, "weight": weight})
            comp_form.setdefault((comp, home), []).append({"pts": home_pts, "weight": weight})
            comp_form.setdefault((comp, away), []).append({"pts": away_pts, "weight": weight})

        self.form_ppg = {}
        for t, entries in form.items():
            recent = entries[-n:]
            if not recent:
                self.form_ppg[t] = 0.5
                continue
            pts = np.array([e["pts"] for e in recent], dtype=float)
            weights = np.array([e["weight"] for e in recent], dtype=float)
            avg_pts = _weighted_avg(pts, weights)
            self.form_ppg[t] = avg_pts / 3.0

        # Form within each competition (how a club played in that tournament).
        self.comp_form_ppg = {}
        for (comp, t), entries in comp_form.items():
            recent = entries[-comp_n:]
            pts = np.array([e["pts"] for e in recent], dtype=float)
            weights = np.array([e["weight"] for e in recent], dtype=float)
            self.comp_form_ppg[f"{comp}|{t}"] = _weighted_avg(pts, weights) / 3.0

    def _form_factor(self, team: str, competition: Optional[str] = None) -> float:
        overall = self.form_ppg.get(team, 0.5)
        comp_ppg = self.comp_form_ppg.get(f"{competition}|{team}") if competition else None
        # Blend same-tournament form with all-competition form when available.
        ppg = overall if comp_ppg is None else 0.4 * overall + 0.6 * comp_ppg
        # Scale: +10% if perfect form, -10% if zero form
        return 1.0 + 0.2 * (ppg - 0.5)

    @staticmethod
    def _context_factor(ctx: Optional[Dict[str, Any]]) -> float:
        """Adjust for a team's standing in its own league log."""
        if not ctx:
            return 1.0
        factor = 1.0
        pos = ctx.get("position")
        size = ctx.get("league_size")
        if pos and size and size >= 5:
            strength = 1.0 - (pos - 1) / (size - 1)  # 1.0 top .. 0.0 bottom
            factor *= 1.0 + POSITION_WEIGHT * (strength - 0.5)
        ppg = ctx.get("ppg")
        if ppg is not None:
            factor *= 1.0 + PPG_WEIGHT * (ppg / 3.0 - 0.5)
        return float(min(CONTEXT_FACTOR_MAX, max(CONTEXT_FACTOR_MIN, factor)))

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        competition: Optional[str] = None,
        contexts: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        home_team = self.resolve_name(home_team)
        away_team = self.resolve_name(away_team)
        if not self.fitted or home_team not in self.teams or away_team not in self.teams:
            return self._fallback(home_team, away_team)

        contexts = contexts or {}
        home_ctx = contexts.get(home_team)
        away_ctx = contexts.get(away_team)

        # Recent form (blended with same-tournament form) x log-position context.
        home_factor = self._form_factor(home_team, competition) * self._context_factor(home_ctx)
        away_factor = self._form_factor(away_team, competition) * self._context_factor(away_ctx)
        home_factor = min(1.3, max(0.75, home_factor))
        away_factor = min(1.3, max(0.75, away_factor))

        home_attack = self.home_attack.get(home_team, 1.0) * home_factor
        home_defense = self.home_defense.get(home_team, 1.0) / home_factor
        away_attack = self.away_attack.get(away_team, 1.0) * away_factor
        away_defense = self.away_defense.get(away_team, 1.0) / away_factor

        ha = 1.0 if neutral else self.home_advantage
        ah = 1.0 / ha

        home_exp = max(
            0.1,
            min(MAX_EXPECTED_GOALS, self.league_avg_home * home_attack * away_defense * ha),
        )
        away_exp = max(
            0.1,
            min(MAX_EXPECTED_GOALS, self.league_avg_away * away_attack * home_defense * ah),
        )

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

        # Blend Elo and Poisson win/draw/loss
        home_win = 0.5 * poisson_home_win + 0.5 * elo_home_win
        draw = 0.5 * poisson_draw + 0.5 * draw
        away_win = 0.5 * poisson_away_win + 0.5 * elo_away_win
        total = home_win + draw + away_win
        if total:
            home_win /= total
            draw /= total
            away_win /= total

        # Show the likeliest scoreline consistent with the predicted outcome —
        # the raw modal score is nearly always 1-1 even for lopsided fixtures.
        best_score, max_prob = self._modal_score(matrix, home_win, draw, away_win)

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
            "model_inputs": {
                "home": self._describe_inputs(home_team, competition, home_ctx),
                "away": self._describe_inputs(away_team, competition, away_ctx),
            },
            "score_matrix": np.array(matrix),
            "last_trained": self.last_trained,
        }

    @staticmethod
    def _modal_score(
        matrix: List[List[float]], home_win: float, draw: float, away_win: float
    ) -> Tuple[str, float]:
        """Most likely scoreline consistent with the predicted outcome."""
        n = len(matrix)
        if home_win >= draw and home_win >= away_win:
            cells = ((i, j) for i in range(n) for j in range(n) if i > j)
        elif away_win >= draw:
            cells = ((i, j) for i in range(n) for j in range(n) if i < j)
        else:
            cells = ((i, j) for i in range(n) for j in range(n) if i == j)
        best, best_score = 0.0, "1-1"
        for i, j in cells:
            if matrix[i][j] > best:
                best, best_score = matrix[i][j], f"{i}-{j}"
        return best_score, best

    def _describe_inputs(
        self,
        team: str,
        competition: Optional[str],
        ctx: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarise the factors that went into a team's prediction."""
        comp_key = f"{competition}|{team}"
        return {
            "form_ppg": round(self.form_ppg.get(team, 0.5) * 3.0, 2),
            "competition": competition,
            "comp_form_ppg": (
                round(self.comp_form_ppg[comp_key] * 3.0, 2)
                if comp_key in self.comp_form_ppg
                else None
            ),
            "position": ctx.get("position") if ctx else None,
            "league_size": ctx.get("league_size") if ctx else None,
            "table_ppg": ctx.get("ppg") if ctx else None,
            "context_league": ctx.get("competition") if ctx else None,
        }

    def _fallback(self, home_team: str, away_team: str) -> Dict[str, Any]:
        home_elo = self.elo.get(home_team, 1500)
        away_elo = self.elo.get(away_team, 1500)
        expected_home = _expected_score(home_elo, away_elo)
        draw = 0.25
        remaining = 1.0 - draw
        home_win = expected_home * remaining
        away_win = (1.0 - expected_home) * remaining
        home_exp = max(0.5, min(4.0, 2.5 * (home_elo / (home_elo + away_elo)) * 1.1))
        away_exp = max(0.5, min(4.0, 2.5 * (away_elo / (home_elo + away_elo))))

        matrix = [
            [_poisson_pmf(i, home_exp) * _poisson_pmf(j, away_exp) for j in range(7)]
            for i in range(7)
        ]
        best_score, max_prob = self._modal_score(matrix, home_win, draw, away_win)
        over = sum(matrix[i][j] for i in range(7) for j in range(7) if i + j > 2.5)
        btts_yes = sum(matrix[i][j] for i in range(1, 7) for j in range(1, 7))

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
                "score": best_score,
                "probability": round(max_prob, 4),
            },
            "over_under_2_5": {"over": round(over, 4), "under": round(1.0 - over, 4)},
            "btts": {"yes": round(btts_yes, 4), "no": round(1.0 - btts_yes, 4)},
            "team_attack_params": {"home": 1.0, "away": 1.0},
            "team_defense_params": {"home": 1.0, "away": 1.0},
            "model_inputs": {
                "home": {"form_ppg": None, "competition": None, "comp_form_ppg": None,
                         "position": None, "league_size": None, "table_ppg": None,
                         "context_league": None},
                "away": {"form_ppg": None, "competition": None, "comp_form_ppg": None,
                         "position": None, "league_size": None, "table_ppg": None,
                         "context_league": None},
            },
            "score_matrix": np.array(matrix),
            "last_trained": self.last_trained,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted parameters to a JSON-safe dict."""
        return {
            "fitted": self.fitted,
            "last_trained": self.last_trained,
            "league_avg_home": self.league_avg_home,
            "league_avg_away": self.league_avg_away,
            "home_advantage": self.home_advantage,
            "home_attack": self.home_attack,
            "home_defense": self.home_defense,
            "away_attack": self.away_attack,
            "away_defense": self.away_defense,
            "elo": self.elo,
            "form_ppg": self.form_ppg,
            "comp_form_ppg": self.comp_form_ppg,
            "name_index": self.name_index,
            "teams": sorted(self.teams),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionEngine":
        """Restore a fitted engine from a serialized dict."""
        engine = cls()
        engine.fitted = data.get("fitted", False)
        engine.last_trained = data.get("last_trained")
        engine.league_avg_home = data.get("league_avg_home", 1.4)
        engine.league_avg_away = data.get("league_avg_away", 1.1)
        engine.home_advantage = data.get("home_advantage", 1.15)
        engine.home_attack = data.get("home_attack", {})
        engine.home_defense = data.get("home_defense", {})
        engine.away_attack = data.get("away_attack", {})
        engine.away_defense = data.get("away_defense", {})
        engine.elo = data.get("elo", {})
        engine.form_ppg = data.get("form_ppg", {})
        engine.comp_form_ppg = data.get("comp_form_ppg", {})
        engine.name_index = data.get("name_index", {})
        engine.teams = set(data.get("teams", []))
        return engine
