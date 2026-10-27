"""Lightweight trainable prediction engine for league matches."""

import logging
import math
import re
import unicodedata
from collections import Counter
from datetime import date, datetime
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


def _rounded_probs(*probs: float) -> Tuple[float, ...]:
    """Round probabilities to 4dp keeping the sum exactly 1.0 — the
    residual goes to the largest component."""
    rounded = [round(p, 4) for p in probs]
    resid = round(1.0 - sum(rounded), 4)
    if resid:
        rounded[rounded.index(max(rounded))] += resid
    return tuple(rounded)


def _draw_prob(elo_diff: float, base_rate: float = 0.25) -> float:
    """Draw probability: learned league draw rate scaled down as the
    favourite's Elo edge grows. `base_rate` comes from fit(); 0.25 is the
    prior used before any data is available."""
    return max(DRAW_FLOOR, base_rate * (1.0 - abs(elo_diff) / DRAW_ELO_SCALE))


# Values below were adopted by scripts/tune.py (see tune_out.txt).
RECENCY_HALF_LIFE_DAYS = 120.0

# Shrink per-team attack/defense ratings toward the league mean when a team
# has few matches — otherwise small samples produce unrealistic xG.
ATTACK_SHRINK_GAMES = 8.0
MAX_EXPECTED_GOALS = 5.0

# Draw model: learned league draw rate (bounded), shrunk as the Elo gap
# grows, with a hard floor so tight low-scoring matchups keep a sane draw
# probability.
DRAW_RATE_MIN = 0.15
DRAW_RATE_MAX = 0.30
DRAW_ELO_SCALE = 1200.0
DRAW_FLOOR = 0.18

# Outcome blend: Poisson goal model vs Elo win expectancy.
BLEND_POISSON = 0.50

# Elo update strength and home-field rating bump.
ELO_K = 30.0
ELO_HOME_FIELD = 70.0

# Recent-form adjustment range: factor applied to attack/defense ratings.
FORM_WEIGHT = 0.2
FORM_FACTOR_MIN = 0.75
FORM_FACTOR_MAX = 1.3

# Dixon-Coles: bound on the low-score correlation parameter fitted from the
# league's draw rate.
DC_RHO_MAX = 0.25

# A competition needs at least this weighted-games mass before its scoring
# baseline stands alone — otherwise shrink toward the pooled mean.
COMP_AVG_SHRINK_GAMES = 20.0
# Matches outside the engine's primary competition move Elo slower — a club
# feasting on weak domestic opponents shouldn't inflate its CL rating.
CROSS_COMP_ELO_K_SCALE = 0.5
# Rating assigned to teams with no history and no prior — below the pool
# mean, since genuinely new/promoted sides tend to be weaker than average.
UNRATED_ELO_MARGIN = 30.0

# Iterations of the opponent-adjustment loop for attack/defense ratings —
# each pass re-measures a team's scoring against the opponents' fitted
# strength instead of raw goals. Two to three passes converge well enough.
OPPONENT_ADJUST_ITERS = 3
# Clamp on the opponent rating used as a divisor so a tiny-sample rating
# can't explode an adjustment.
OPPONENT_RATING_CLIP = (0.35, 3.0)

# Same-fixture duplicate window: rows for the same canonical home/away pair
# within this span are one fixture reported by two feeds.
DUPLICATE_WINDOW_SECONDS = 26 * 3600

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


# Letters NFKD won't decompose — transliterated first so feeds spelling
# "Bodø/Glimt" vs "Bodo/Glimt" or "Fenerbahçe" vs "Fenerbahce" still
# produce the same key.
_TEAM_TRANSLIT = str.maketrans(
    {
        "ø": "o", "æ": "ae", "œ": "oe", "ð": "d", "þ": "th",
        "ł": "l", "ß": "ss", "đ": "d", "ı": "i", "ħ": "h",
        "Ø": "O", "Æ": "AE", "Œ": "OE", "Ð": "D", "Þ": "TH",
        "Ł": "L", "Đ": "D", "Ħ": "H", "İ": "I",
    }
)


def _norm_key(name: str) -> str:
    """Normalise a team name for cross-source / cross-competition matching."""
    decomposed = unicodedata.normalize("NFKD", name.translate(_TEAM_TRANSLIT))
    ascii_name = "".join(c for c in decomposed if not unicodedata.combining(c))
    tokens = re.sub(r"[^a-z0-9 ]", " ", ascii_name.lower()).split()
    tokens = [t for t in tokens if t not in TEAM_STOP_TOKENS and not t.isdigit()]
    return " ".join(tokens) or name.lower().strip()


# Whole-name aliases for recurring cross-source spellings that token
# matching cannot bridge, e.g. "Man Utd" vs "Manchester United".
_TEAM_KEY_ALIASES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "spurs": "tottenham hotspur",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "wolves": "wolverhampton wanderers",
    "psv eindhoven": "psv",
    "sporting lisbon": "sporting clube portugal",
    "sporting cp": "sporting clube portugal",
    "m gladbach": "monchengladbach",
    "borussia m gladbach": "borussia monchengladbach",
    "ivory coast": "cote d ivoire",
    "turkey": "turkiye",
    "south korea": "korea republic",
    "korea south": "korea republic",
    "czech republic": "czechia",
    "cape verde": "cabo verde",
    "swaziland": "eswatini",
    "holland": "netherlands",
    "dr congo": "congo dr",
    "democratic republic of congo": "congo dr",
}

# Markers that distinguish otherwise identical names — a "Germany U21"
# fixture must never merge into a senior "Germany" one.
_TEAM_QUALIFIER_RE = re.compile(
    r"\b(?:u-?\s?\d+|women|woman|ladies|femenino|femminile|feminine|frauen|"
    r"reserves?|academy|youth|olympic|esports?)\b",
    re.IGNORECASE,
)


def _team_qualifiers(name: str) -> frozenset:
    return frozenset(
        re.sub(r"[\s-]", "", m.group(0).lower())
        for m in _TEAM_QUALIFIER_RE.finditer(name)
    )


def _covered(token: str, others: set) -> bool:
    if token in others:
        return True
    if len(token) == 1:
        # Initials: "B. Dortmund" vs "Borussia Dortmund".
        return any(o.startswith(token) for o in others)
    if len(token) == 3:
        # Short prefixes: "Man" vs "Manchester".
        return any(o.startswith(token) for o in others)
    if len(token) >= 4:
        # Suffixes: "Gladbach" vs "Monchengladbach". Prefixes this long are
        # too loose ("Roma" vs "Romania", "Villa" vs "Villarreal").
        return any(o.endswith(token) for o in others)
    return False


def teams_equivalent(a: str, b: str) -> bool:
    """True when two feed strings almost certainly denote the same team."""
    if not a or not b:
        return False
    if a.strip().lower() == b.strip().lower():
        return True
    ka = _TEAM_KEY_ALIASES.get(_norm_key(a), _norm_key(a))
    kb = _TEAM_KEY_ALIASES.get(_norm_key(b), _norm_key(b))
    if ka == kb:
        return True
    if _team_qualifiers(a) != _team_qualifiers(b):
        return False
    ta, tb = ka.split(), kb.split()
    if not ta or not tb:
        return False
    # Acronyms: "PSG" vs "Paris Saint-Germain".
    if len(ta) == 1 and ta[0] == "".join(t[0] for t in tb):
        return True
    if len(tb) == 1 and tb[0] == "".join(t[0] for t in ta):
        return True
    # The shorter token list must expand fully into the longer one; with
    # equal lengths either side may hold the abbreviation ("B. Dortmund"
    # vs "Borussia Dortmund"), so try both directions.
    sa, sb = set(ta), set(tb)
    if len(sa) <= len(sb) and all(_covered(t, sb) for t in sa):
        return True
    return len(sb) <= len(sa) and all(_covered(t, sa) for t in sb)


def _fixture_dt(m: Dict[str, Any]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(m.get("date") or "")
    except (ValueError, TypeError):
        return None


def fixtures_equivalent(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True when two match rows almost certainly describe the same fixture.

    Feeds disagree on team spelling ("Man City" vs "Manchester City") and
    kickoff times drift by minutes or sit either side of a timezone
    midnight, so pair on team equivalence plus a tolerant date window.
    """
    if not (
        teams_equivalent(a.get("home_team", ""), b.get("home_team", ""))
        and teams_equivalent(a.get("away_team", ""), b.get("away_team", ""))
    ):
        return False
    exact = _norm_key(a.get("home_team", "")) == _norm_key(
        b.get("home_team", "")
    ) and _norm_key(a.get("away_team", "")) == _norm_key(b.get("away_team", ""))
    da, db = _fixture_dt(a), _fixture_dt(b)
    if da and db and "T" in (a.get("date") or "") and "T" in (b.get("date") or ""):
        limit = 26 * 3600 if exact else 2 * 3600
        return abs((da - db).total_seconds()) <= limit
    day_a, day_b = (a.get("date") or "")[:10], (b.get("date") or "")[:10]
    if not day_a or not day_b:
        return exact
    if day_a == day_b:
        return True
    if not exact:
        return False
    try:
        # Date-only feeds can land a day off the real fixture date.
        return abs((date.fromisoformat(day_a) - date.fromisoformat(day_b)).days) <= 1
    except ValueError:
        return False


def _weighted_avg(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def _apply_dixon_coles(
    matrix: List[List[float]], home_exp: float, away_exp: float, rho: float
) -> None:
    """Reweight the four lowest-score cells by the DC tau correction and
    renormalise the matrix to sum to 1. rho < 0 boosts draws."""
    if not rho:
        return
    tau = (
        (0, 0, 1.0 - home_exp * away_exp * rho),
        (0, 1, 1.0 + home_exp * rho),
        (1, 0, 1.0 + away_exp * rho),
        (1, 1, 1.0 - rho),
    )
    for i, j, t in tau:
        matrix[i][j] = max(0.0, matrix[i][j] * t)
    total = sum(sum(row) for row in matrix)
    if total > 0:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] /= total


class PredictionEngine:
    """Fits attack/defense, Elo and form from match history then predicts."""

    def __init__(self):
        self.fitted = False
        self.last_trained: Optional[str] = None

        self.league_avg_home: float = 1.4
        self.league_avg_away: float = 1.1
        self.home_advantage: float = 1.15
        self.draw_rate: float = 0.25

        self.home_attack: Dict[str, float] = {}
        self.home_defense: Dict[str, float] = {}
        self.away_attack: Dict[str, float] = {}
        self.away_defense: Dict[str, float] = {}
        self.elo: Dict[str, float] = {}
        # Normalised-name Elo priors (e.g. national rankings) — lets the
        # fallback path rate teams that never appear in the match pool.
        self.elo_prior: Dict[str, float] = {}
        # Rating assumed for teams with neither history nor a prior.
        self.elo_unrated: float = 1500.0 - UNRATED_ELO_MARGIN
        # Per-competition scoring baselines: competition -> (avg home, avg away).
        self.comp_avg: Dict[str, Tuple[float, float]] = {}
        # Per-competition draw rates, shrunk toward the pooled rate.
        self.comp_draw_rate: Dict[str, float] = {}
        # Per-competition Dixon-Coles low-score correlation, fitted from
        # each competition's own draw rate and scoring baseline.
        self.comp_rho: Dict[str, float] = {}
        # Dixon-Coles low-score correlation fitted from the draw rate.
        self.dc_rho: float = 0.0
        self._primary_competition: Optional[str] = None
        self.form_ppg: Dict[str, float] = {}
        self.comp_form_ppg: Dict[str, float] = {}
        self.name_index: Dict[str, str] = {}
        self.teams: set = set()
        # Freshness metadata — how much data the model saw and how recent
        # the newest result in the primary competition is.
        self.matches_used: int = 0
        self.data_as_of: Optional[str] = None

    def fit(
        self,
        matches: List[Dict[str, Any]],
        initial_elo: Optional[Dict[str, float]] = None,
        primary_competition: Optional[str] = None,
    ) -> None:
        """Train on a list of full-time match dicts. `initial_elo` seeds
        starting ratings (e.g. national-team rankings) instead of 1500.
        `primary_competition` is the engine's home competition — matches
        from related competitions are attenuated in the Elo update and the
        scoring baselines are kept per-competition."""
        self._initial_elo = initial_elo or {}
        self._primary_competition = primary_competition
        self.elo_prior = {
            _TEAM_KEY_ALIASES.get(_norm_key(k), _norm_key(k)): float(v)
            for k, v in self._initial_elo.items()
        }
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

        # Drop duplicate rows for the same fixture — the training pool mixes
        # current, historical and related-competition pulls that can overlap.
        # After canonicalisation the same fixture has identical team names,
        # so a tolerant date window catches timezone/day drift across feeds.
        df = df.sort_values("date")
        pair = df["home_team"] + "|" + df["away_team"]
        dup = (
            df.groupby(pair)["date"].diff().dt.total_seconds().abs()
            <= DUPLICATE_WINDOW_SECONDS
        ).fillna(False)
        if dup.any():
            df = df.loc[~dup]

        self.teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

        # Recency weighting: more recent matches count more. _fit_core
        # recomputes it relative to each walk-forward cut date.
        latest = df["date"].max()
        df["days_ago"] = (latest - df["date"]).dt.days.astype(float)
        df["weight"] = np.exp(-np.log(2) / RECENCY_HALF_LIFE_DAYS * df["days_ago"])
        df["weight"] = df["weight"].clip(lower=0.01)

        self._fit_core(df)

        self.fitted = True
        self.last_trained = datetime.now().isoformat()
        self.matches_used = int(len(df))
        if primary_competition is not None:
            primary = df[df["competition"].astype(str) == str(primary_competition)]
        else:
            primary = df
        newest = primary["date"].max() if not primary.empty else df["date"].max()
        self.data_as_of = newest.isoformat() if pd.notna(newest) else None
        logger.info(f"PredictionEngine trained on {len(df)} matches")

    def _fit_core(self, df: pd.DataFrame) -> None:
        """Fit every model parameter on a canonicalised match frame
        (team names resolved, `date`/`competition` columns present)."""
        df = df.copy()
        # Teams present in this frame — also keeps inner walk-forward
        # engines (which skip fit()) consistent.
        self.teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

        # Recency weighting: more recent matches count more.
        latest = df["date"].max()
        df["days_ago"] = (latest - df["date"]).dt.days.astype(float)
        df["weight"] = np.exp(-np.log(2) / RECENCY_HALF_LIFE_DAYS * df["days_ago"])
        df["weight"] = df["weight"].clip(lower=0.01)
        weights = df["weight"].values

        # Per-competition scoring baselines, shrunk toward the pooled mean —
        # a mixed multi-league pool (e.g. CL + five domestic leagues) no
        # longer shares one global goal average.
        global_home = max(_weighted_avg(df["home_score"].values, weights), 0.1)
        global_away = max(_weighted_avg(df["away_score"].values, weights), 0.1)
        self.comp_avg = {}
        for comp, grp in df.groupby("competition"):
            w = grp["weight"].values
            wsum = float(w.sum())
            ch = _weighted_avg(grp["home_score"].values, w)
            ca = _weighted_avg(grp["away_score"].values, w)
            sh = (ch * wsum + COMP_AVG_SHRINK_GAMES * global_home) / (
                wsum + COMP_AVG_SHRINK_GAMES
            )
            sa = (ca * wsum + COMP_AVG_SHRINK_GAMES * global_away) / (
                wsum + COMP_AVG_SHRINK_GAMES
            )
            self.comp_avg[str(comp)] = (max(sh, 0.1), max(sa, 0.1))

        # Home advantage belongs to the primary competition, falling back
        # to the pooled baselines when it has no scoring baseline of its own.
        pc = self._primary_competition
        self.league_avg_home, self.league_avg_away = self.comp_avg.get(
            str(pc), (global_home, global_away)
        )
        self.home_advantage = max(1.05, min(1.4, self.league_avg_home / self.league_avg_away))

        # Draw rate learned per competition — low-scoring leagues draw
        # more — shrunk toward the pooled rate for thin samples, with a
        # matching per-competition Dixon-Coles correlation.
        global_draw = _weighted_avg(
            (df["home_score"] == df["away_score"]).astype(float).values, weights
        )
        self.comp_draw_rate = {}
        self.comp_rho = {}
        for comp, grp in df.groupby("competition"):
            w = grp["weight"].values
            wsum = float(w.sum())
            d = _weighted_avg(
                (grp["home_score"] == grp["away_score"]).astype(float).values, w
            )
            rate = (d * wsum + COMP_AVG_SHRINK_GAMES * global_draw) / (
                wsum + COMP_AVG_SHRINK_GAMES
            )
            rate = float(min(DRAW_RATE_MAX, max(DRAW_RATE_MIN, rate)))
            self.comp_draw_rate[str(comp)] = rate
            ch, ca = self.comp_avg[str(comp)]
            self.comp_rho[str(comp)] = self._dc_rho_for(rate, ch, ca)
        self.draw_rate = self.comp_draw_rate.get(
            str(pc), float(min(DRAW_RATE_MAX, max(DRAW_RATE_MIN, global_draw)))
        )
        self.dc_rho = self.comp_rho.get(str(pc), 0.0)

        # Attack/defense ratings on competition-normalised scores — a goal
        # is measured relative to the environment it was scored in, then
        # converted to the predicted competition's baseline at predict time.
        comp_h = df["competition"].map(lambda c: self.comp_avg[str(c)][0])
        comp_a = df["competition"].map(lambda c: self.comp_avg[str(c)][1])
        df["nh"] = df["home_score"] / comp_h
        df["na"] = df["away_score"] / comp_a

        # Opponent-adjusted ratings: a team's scoring is measured against the
        # *fitted* strength of the opponents it faced, not raw goals — a soft
        # early schedule no longer inflates an attack rating. Each pass
        # re-solves one rating holding the others fixed; a few iterations
        # converge well enough for this additive model. Ratings are still
        # shrunk toward the mean for small samples.
        weight = df["weight"]

        def _shrunk_ratios(keys: pd.Series, values: pd.Series) -> pd.Series:
            num = (values * weight).groupby(keys).sum()
            den = weight.groupby(keys).sum()
            raw = num / den
            return (raw * den + ATTACK_SHRINK_GAMES) / (den + ATTACK_SHRINK_GAMES)

        teams_idx = sorted(self.teams)
        home_attack = pd.Series(1.0, index=teams_idx)
        home_defense = pd.Series(1.0, index=teams_idx)
        away_attack = pd.Series(1.0, index=teams_idx)
        away_defense = pd.Series(1.0, index=teams_idx)

        # Contribution semantics mirror predict(): home expected goals are
        # avg_h * home_attack[h] * away_defense[a] * ha, so each match's
        # contribution to a rating is the normalised score divided by the
        # opposing rating it was scored against.
        lo, hi = OPPONENT_RATING_CLIP
        for _ in range(OPPONENT_ADJUST_ITERS):
            opp_away_def = df["away_team"].map(away_defense).fillna(1.0).clip(lo, hi)
            home_attack = _shrunk_ratios(df["home_team"], df["nh"] / opp_away_def)
            opp_home_def = df["home_team"].map(home_defense).fillna(1.0).clip(lo, hi)
            away_attack = _shrunk_ratios(df["away_team"], df["na"] / opp_home_def)
            opp_away_att = df["away_team"].map(away_attack).fillna(1.0).clip(lo, hi)
            home_defense = _shrunk_ratios(df["home_team"], df["na"] / opp_away_att)
            opp_home_att = df["home_team"].map(home_attack).fillna(1.0).clip(lo, hi)
            away_defense = _shrunk_ratios(df["away_team"], df["nh"] / opp_home_att)

        self.home_attack = home_attack.to_dict()
        self.home_defense = home_defense.to_dict()
        self.away_attack = away_attack.to_dict()
        self.away_defense = away_defense.to_dict()

        # Elo ratings
        self._fit_elo(df)
        self.elo_unrated = (
            float(np.mean(list(self.elo.values()))) - UNRATED_ELO_MARGIN
            if self.elo
            else 1500.0 - UNRATED_ELO_MARGIN
        )

        # Recent form (weighted last 10 matches)
        self._fit_form(df, n=10)

    @staticmethod
    def _dc_rho_for(draw_rate: float, lam: float, mu: float) -> float:
        """Dixon-Coles low-score correlation implied by the gap between a
        competition's observed draw rate and what independent Poisson
        would predict at its average scoring rates."""
        p00 = _poisson_pmf(0, lam) * _poisson_pmf(0, mu)
        p11 = _poisson_pmf(1, lam) * _poisson_pmf(1, mu)
        rest = sum(_poisson_pmf(i, lam) * _poisson_pmf(i, mu) for i in range(2, 12))
        indep = p00 + p11 + rest
        denom = lam * mu * p00 + p11
        rho = (indep - draw_rate) / denom if denom > 0 else 0.0
        return max(-DC_RHO_MAX, min(DC_RHO_MAX, rho))

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
        for k in (key, _TEAM_KEY_ALIASES.get(key)):
            if k and k in self.name_index:
                return self.name_index[k]
        # Last resort: equivalence rules over the known team set, so feed
        # spellings like "Inter" still reach "FC Internazionale Milano"
        # instead of silently degrading to the weak fallback path.
        for t in self.teams:
            if teams_equivalent(name, t):
                return t
        return name

    def _fit_elo(self, df: pd.DataFrame, k: float = ELO_K, home_field: float = ELO_HOME_FIELD):
        priors = getattr(self, "_initial_elo", {})
        # Priors are matched by exact name first, then by normalised key so
        # a seed entry like "Atleti" still reaches a feed's "Atlético Madrid".
        norm_priors = {
            _TEAM_KEY_ALIASES.get(_norm_key(k2), _norm_key(k2)): float(v)
            for k2, v in priors.items()
        }
        self.elo = {
            t: float(
                priors.get(
                    t,
                    norm_priors.get(
                        _TEAM_KEY_ALIASES.get(_norm_key(t), _norm_key(t)),
                        1500.0,
                    ),
                )
            )
            for t in self.teams
        }
        for row in df.sort_values("date").itertuples():
            home = row.home_team
            away = row.away_team
            hs, aws = row.home_score, row.away_score
            weight = getattr(row, "weight", 1.0)

            expected = _expected_score(self.elo[home] + home_field, self.elo[away])
            if hs > aws:
                actual = 1.0
            elif hs == aws:
                actual = 0.5
            else:
                actual = 0.0

            # Scale k by the winning margin — a 5-0 moves ratings ~2.6x more
            # than a 1-0; draws use 0.6k so they still shift ratings a little.
            margin = abs(hs - aws)
            k_eff = k * min(4.0, max(0.6, math.log2(margin + 1)))
            # Cross-competition matches inform strength but move ratings
            # slower — a club dominating a weak domestic league shouldn't
            # overrate into its Champions League prediction.
            comp = getattr(row, "competition", None)
            if self._primary_competition and comp != self._primary_competition:
                k_eff *= CROSS_COMP_ELO_K_SCALE
            delta = k_eff * weight * (actual - expected)
            self.elo[home] += delta
            self.elo[away] -= delta

        # Ensure every team has a default rating
        for t in self.teams:
            self.elo.setdefault(t, 1500.0)

    def _fit_form(self, df: pd.DataFrame, n: int = 10, comp_n: int = 6):
        form: Dict[str, List[Dict[str, Any]]] = {t: [] for t in self.teams}
        comp_form: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in df.sort_values("date").itertuples():
            home, away = row.home_team, row.away_team
            comp = getattr(row, "competition", None) or "default"
            hs, aws = row.home_score, row.away_score
            if hs > aws:
                home_pts, away_pts = 3, 0
            elif hs == aws:
                home_pts, away_pts = 1, 1
            else:
                home_pts, away_pts = 0, 3
            weight = getattr(row, "weight", 1.0)
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
        return 1.0 + FORM_WEIGHT * (ppg - 0.5)

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
        home_factor = min(FORM_FACTOR_MAX, max(FORM_FACTOR_MIN, home_factor))
        away_factor = min(FORM_FACTOR_MAX, max(FORM_FACTOR_MIN, away_factor))

        home_attack = self.home_attack.get(home_team, 1.0) * home_factor
        home_defense = self.home_defense.get(home_team, 1.0) / home_factor
        away_attack = self.away_attack.get(away_team, 1.0) * away_factor
        away_defense = self.away_defense.get(away_team, 1.0) / away_factor

        ha = 1.0 if neutral else self.home_advantage
        ah = 1.0 / ha

        # Convert competition-normalised ratings to the target
        # competition's scoring baseline.
        avg_h, avg_a = self.comp_avg.get(
            str(competition), (self.league_avg_home, self.league_avg_away)
        )
        home_exp = max(
            0.1,
            min(MAX_EXPECTED_GOALS, avg_h * home_attack * away_defense * ha),
        )
        away_exp = max(
            0.1,
            min(MAX_EXPECTED_GOALS, avg_a * away_attack * home_defense * ah),
        )

        # Elo-based outcome probabilities
        field = 0.0 if neutral else ELO_HOME_FIELD
        elo_diff = self.elo[home_team] - self.elo[away_team] + field
        elo_win = _expected_score(self.elo[home_team], self.elo[away_team], field)
        draw_rate = self.comp_draw_rate.get(str(competition), self.draw_rate)
        draw = _draw_prob(elo_diff, draw_rate)
        remaining = 1.0 - draw
        elo_home_win = elo_win * remaining
        elo_away_win = (1.0 - elo_win) * remaining

        # Poisson-based outcome and score matrix, with the Dixon-Coles
        # low-score correction applied before outcomes are tallied.
        max_goals = 6
        matrix = [
            [
                _poisson_pmf(i, home_exp) * _poisson_pmf(j, away_exp)
                for j in range(max_goals + 1)
            ]
            for i in range(max_goals + 1)
        ]
        rho = self.comp_rho.get(str(competition), self.dc_rho)
        _apply_dixon_coles(matrix, home_exp, away_exp, rho)
        poisson_home_win = poisson_draw = poisson_away_win = 0.0
        over = under = btts_yes = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = matrix[i][j]
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
        wp = BLEND_POISSON
        home_win = wp * poisson_home_win + (1 - wp) * elo_home_win
        draw = wp * poisson_draw + (1 - wp) * draw
        away_win = wp * poisson_away_win + (1 - wp) * elo_away_win
        total = home_win + draw + away_win
        if total:
            home_win /= total
            draw /= total
            away_win /= total
        home_win, draw, away_win = _rounded_probs(home_win, draw, away_win)

        # Show the likeliest scoreline consistent with the predicted outcome —
        # the raw modal score is nearly always 1-1 even for lopsided fixtures.
        best_score, max_prob = self._modal_score(matrix, home_win, draw, away_win)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "model": "trained_poisson_elo",
            "expected_goals": {"home": round(home_exp, 2), "away": round(away_exp, 2)},
            "outcome_probabilities": {
                "home_win": home_win,
                "draw": draw,
                "away_win": away_win,
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
            "data_as_of": self.data_as_of,
            "matches_used": self.matches_used,
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

    def _team_elo(self, team: str) -> float:
        """Best available rating: trained Elo, then normalised-name prior,
        then a below-mean default for unseen (e.g. promoted) teams."""
        elo = self.elo.get(team)
        if elo is not None:
            return elo
        key = _norm_key(team)
        key = _TEAM_KEY_ALIASES.get(key, key)
        return self.elo_prior.get(key, self.elo_unrated)

    def _fallback(self, home_team: str, away_team: str) -> Dict[str, Any]:
        home_elo = self._team_elo(home_team)
        away_elo = self._team_elo(away_team)
        expected_home = _expected_score(home_elo, away_elo)
        draw = max(DRAW_FLOOR, self.draw_rate)
        remaining = 1.0 - draw
        home_win = expected_home * remaining
        away_win = (1.0 - expected_home) * remaining
        home_win, draw, away_win = _rounded_probs(home_win, draw, away_win)
        home_exp = max(0.5, min(4.0, 2.5 * (home_elo / (home_elo + away_elo)) * 1.1))
        away_exp = max(0.5, min(4.0, 2.5 * (away_elo / (home_elo + away_elo))))

        matrix = [
            [_poisson_pmf(i, home_exp) * _poisson_pmf(j, away_exp) for j in range(7)]
            for i in range(7)
        ]
        _apply_dixon_coles(matrix, home_exp, away_exp, self.dc_rho)
        best_score, max_prob = self._modal_score(matrix, home_win, draw, away_win)
        over = sum(matrix[i][j] for i in range(7) for j in range(7) if i + j > 2.5)
        btts_yes = sum(matrix[i][j] for i in range(1, 7) for j in range(1, 7))

        return {
            "home_team": home_team,
            "away_team": away_team,
            "model": "elo_fallback",
            "expected_goals": {"home": round(home_exp, 2), "away": round(away_exp, 2)},
            "outcome_probabilities": {
                "home_win": home_win,
                "draw": draw,
                "away_win": away_win,
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
            "data_as_of": self.data_as_of,
            "matches_used": self.matches_used,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted parameters to a JSON-safe dict."""
        return {
            "fitted": self.fitted,
            "last_trained": self.last_trained,
            "league_avg_home": self.league_avg_home,
            "league_avg_away": self.league_avg_away,
            "home_advantage": self.home_advantage,
            "draw_rate": self.draw_rate,
            "home_attack": self.home_attack,
            "home_defense": self.home_defense,
            "away_attack": self.away_attack,
            "away_defense": self.away_defense,
            "elo": self.elo,
            "elo_prior": self.elo_prior,
            "elo_unrated": self.elo_unrated,
            "comp_avg": {k: list(v) for k, v in self.comp_avg.items()},
            "comp_draw_rate": self.comp_draw_rate,
            "comp_rho": self.comp_rho,
            "dc_rho": self.dc_rho,
            "primary_competition": self._primary_competition,
            "form_ppg": self.form_ppg,
            "comp_form_ppg": self.comp_form_ppg,
            "name_index": self.name_index,
            "teams": sorted(self.teams),
            "matches_used": self.matches_used,
            "data_as_of": self.data_as_of,
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
        engine.draw_rate = data.get("draw_rate", 0.25)
        engine.home_attack = data.get("home_attack", {})
        engine.home_defense = data.get("home_defense", {})
        engine.away_attack = data.get("away_attack", {})
        engine.away_defense = data.get("away_defense", {})
        engine.elo = data.get("elo", {})
        engine.elo_prior = data.get("elo_prior", {})
        engine.elo_unrated = data.get("elo_unrated", 1500.0 - UNRATED_ELO_MARGIN)
        engine.comp_avg = {
            k: tuple(v) for k, v in (data.get("comp_avg") or {}).items()
        }
        engine.comp_draw_rate = data.get("comp_draw_rate", {})
        engine.comp_rho = data.get("comp_rho", {})
        engine.dc_rho = data.get("dc_rho", 0.0)
        engine._primary_competition = data.get("primary_competition")
        engine.form_ppg = data.get("form_ppg", {})
        engine.comp_form_ppg = data.get("comp_form_ppg", {})
        engine.name_index = data.get("name_index", {})
        engine.teams = set(data.get("teams", []))
        engine.matches_used = data.get("matches_used", 0)
        engine.data_as_of = data.get("data_as_of")
        return engine
