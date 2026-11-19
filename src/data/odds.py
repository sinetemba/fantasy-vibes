"""Bookmaker odds — free match CSVs from football-data.co.uk.

Two uses:
- `market_probs` supplies implied pre-match probabilities for upcoming
  fixtures (from fixtures.csv), which LeagueService blends into its
  predictions as a third component alongside Poisson and Elo.
- `benchmark` scores every settled prediction in the log, on the same
  fixtures, against the odds-implied probabilities. The gap between the
  model's log-loss and the market's is the honest measure of how far the
  model sits from the best public forecast — shown on the About page.
"""

import csv
import io
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from .prediction_engine import teams_equivalent
from .utils import cached_get
from . import prediction_log

logger = logging.getLogger(__name__)

# football-data.co.uk division codes for the app's club leagues.
LEAGUE_DIV = {
    "PL": "E0",
    "BL1": "D1",
    "SA": "I1",
    "PD": "SP1",
    "FL1": "F1",
}

_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
# Current-season CSVs update roughly weekly; finished seasons never
# change, but a single TTL plus the on-disk stale fallback keeps it simple.
CACHE_TTL = 24 * 3600

# Upcoming fixtures with pre-match odds across every covered division —
# refreshes daily as the next slate is priced. This is the live source
# for prediction blending; the season CSVs only list played matches.
_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_TTL = 3600

# Odds columns in preference order: market average, then Bet365, then
# Pinnacle closing odds.
_ODDS_COLUMNS = (
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
    ("PSCH", "PSCD", "PSCA"),
)


def _season_code(d: date) -> str:
    """football-data.co.uk season label, e.g. "2526" for 2025-26."""
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _implied(row: Dict[str, Any], cols: Tuple[str, str, str]) -> Optional[Tuple[float, float, float]]:
    """Normalised 1/odds probabilities, or None when a price is missing."""
    try:
        raw = [float(row[c]) for c in cols]
    except (KeyError, TypeError, ValueError):
        return None
    if any(o <= 1.0 for o in raw):
        return None
    inv = [1.0 / o for o in raw]
    total = sum(inv)
    return tuple(p / total for p in inv)


def _parse_date(raw: Optional[str]) -> Optional[date]:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except (ValueError, AttributeError):
            pass
    return None


def _csv_text(text: str) -> str:
    """Normalise a football-data.co.uk CSV to proper Unicode. The files
    are UTF-8 with a BOM but served without a charset, so requests decodes
    them as latin-1 — the BOM survives as mojibake 'ï»¿' and non-ASCII
    names are mangled. Round-trip back to UTF-8 in that case."""
    if text.startswith("ï»¿"):
        try:
            return text.encode("latin-1").decode("utf-8-sig")
        except UnicodeError:
            pass
    return text.lstrip("\ufeff")


def _parse_rows(text: str, div: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse {date, home, away, probs} rows from a football-data.co.uk CSV;
    filtered to one division when `div` is given."""
    rows = []
    for r in csv.DictReader(io.StringIO(_csv_text(text))):
        if div and (r.get("Div") or "").strip() != div:
            continue
        d = _parse_date(r.get("Date"))
        home, away = (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
        if not d or not home or not away:
            continue
        probs = next((p for p in (_implied(r, c) for c in _ODDS_COLUMNS) if p), None)
        if probs:
            rows.append({"date": d, "home": home, "away": away, "probs": probs})
    return rows


def odds_rows(league_code: str, season: str) -> List[Dict[str, Any]]:
    """Parsed {date, home, away, probs} rows for one league-season CSV."""
    div = LEAGUE_DIV.get(league_code)
    if not div:
        return []
    text = cached_get(_URL.format(season=season, div=div), ttl_seconds=CACHE_TTL, raw=True)
    if not text:
        return []
    return _parse_rows(text)


# Parsed-CSV memo on top of the 24h disk cache — prediction passes match
# many fixtures per league-season and shouldn't reparse the CSV each time.
_ROWS_MEMO_TTL = 3600.0
_rows_memo: Dict[Tuple[str, str], Tuple[float, List[Dict[str, Any]]]] = {}
_rows_memo_lock = threading.Lock()


def _rows(league_code: str, season: str) -> List[Dict[str, Any]]:
    key = (league_code, season)
    hit = _rows_memo.get(key)
    if hit and (time.time() - hit[0]) < _ROWS_MEMO_TTL:
        return hit[1]
    rows = odds_rows(league_code, season)
    with _rows_memo_lock:
        _rows_memo[key] = (time.time(), rows)
    return rows


def _fixture_rows(league_code: str) -> List[Dict[str, Any]]:
    """Upcoming fixtures with odds for this league from fixtures.csv —
    the only fd.co.uk file carrying unplayed matches."""
    key = (league_code, "__fixtures__")
    hit = _rows_memo.get(key)
    if hit and (time.time() - hit[0]) < FIXTURES_TTL:
        return hit[1]
    rows = []
    div = LEAGUE_DIV.get(league_code)
    if div:
        text = cached_get(_FIXTURES_URL, ttl_seconds=FIXTURES_TTL, raw=True)
        if text:
            rows = _parse_rows(text, div)
    with _rows_memo_lock:
        _rows_memo[key] = (time.time(), rows)
    return rows


def _fixture_day(when: Any) -> Optional[date]:
    """Fixture match-day in SAST — accepts ISO strings or date/datetime."""
    if isinstance(when, (datetime, date)):
        when = when.isoformat()
    kick = prediction_log.kickoff_sast(str(when or ""))
    return kick.date() if kick else None


def _match_probs(
    home: str, away: str, day: date, rows: List[Dict[str, Any]]
) -> Optional[Tuple[float, float, float]]:
    """Market implied probs for a fixture — same-day match first, then
    ±1 day for feeds disagreeing on the match date."""
    for tolerance in (0, 1):
        for r in rows:
            if abs((r["date"] - day).days) != tolerance:
                continue
            if teams_equivalent(home, r["home"]) and teams_equivalent(away, r["away"]):
                return r["probs"]
    return None


def market_probs(
    league_code: str, home: str, away: str, when: Any
) -> Optional[Tuple[float, float, float]]:
    """Implied pre-match probabilities (home, draw, away) for a fixture,
    or None when the league isn't odds-covered, the date is missing, or
    no feed has priced the fixture. Upcoming fixtures come from
    fixtures.csv; the season CSV is the fallback."""
    if league_code not in LEAGUE_DIV:
        return None
    day = _fixture_day(when)
    if day is None:
        return None
    probs = _match_probs(home, away, day, _fixture_rows(league_code))
    if probs:
        return probs
    return _match_probs(home, away, day, _rows(league_code, _season_code(day)))


def benchmark(days: int = 30) -> Dict[str, Any]:
    """Score settled log predictions against market odds on the same
    fixtures. `n` counts entries matched to an odds row; `total` counts
    settled entries in odds-covered leagues (match coverage)."""
    entries = [
        e for e in prediction_log._settled_entries(days) if e.get("league") in LEAGUE_DIV
    ]
    # Prefetch every needed league-season CSV in parallel — the About page
    # calls this on render and a serial per-league fetch stalls it.
    needed = set()
    for e in entries:
        kick = prediction_log.kickoff_sast(e.get("date"))
        if kick is not None:
            needed.add((e["league"], _season_code(kick.date())))
    fetched: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if needed:
        keys = list(needed)
        with ThreadPoolExecutor(max_workers=min(4, len(keys))) as pool:
            for key, rows in zip(keys, pool.map(lambda k: _rows(*k), keys)):
                fetched[key] = rows

    n = 0
    model_ll = market_ll = model_brier = market_brier = 0.0
    model_hit = market_hit = 0
    # Entries logged since the market blend shipped carry m_* (pre-blend
    # model probs) — p_* is the blended distribution users saw. Scoring
    # p_* against the market it contains is circular, so the model series
    # uses m_* where it exists and the blend is scored on its own subset.
    blend_n = 0
    blend_ll = blend_base_ll = blend_brier = blend_base_brier = 0.0
    blend_hit = blend_base_hit = 0
    for e in entries:
        kick = prediction_log.kickoff_sast(e.get("date"))
        if kick is None:
            continue
        market = _match_probs(
            e["home_team"], e["away_team"], kick.date(),
            fetched.get((e["league"], _season_code(kick.date())), []),
        )
        if market is None:
            continue
        disp = (float(e["p_home"]), float(e["p_draw"]), float(e["p_away"]))
        model = (
            float(e.get("m_home", e["p_home"])),
            float(e.get("m_draw", e["p_draw"])),
            float(e.get("m_away", e["p_away"])),
        )
        idx = prediction_log._RESULT_IDX[e["result"]]
        n += 1
        model_ll += -math.log(max(model[idx], 1e-9))
        market_ll += -math.log(max(market[idx], 1e-9))
        for probs, acc in ((model, "model"), (market, "market")):
            brier = sum((p - (i == idx)) ** 2 for i, p in enumerate(probs))
            if acc == "model":
                model_brier += brier
                model_hit += probs.index(max(probs)) == idx
            else:
                market_brier += brier
                market_hit += probs.index(max(probs)) == idx
        if e.get("m_home") is not None:
            blend_n += 1
            for probs, acc in ((disp, "blend"), (model, "base")):
                ll = -math.log(max(probs[idx], 1e-9))
                brier = sum((p - (i == idx)) ** 2 for i, p in enumerate(probs))
                hit = probs.index(max(probs)) == idx
                if acc == "blend":
                    blend_ll += ll
                    blend_brier += brier
                    blend_hit += hit
                else:
                    blend_base_ll += ll
                    blend_base_brier += brier
                    blend_base_hit += hit
    if not n:
        return {"n": 0, "total": len(entries), "days": days}
    return {
        "n": n,
        "total": len(entries),
        "days": days,
        "model_logloss": model_ll / n,
        "market_logloss": market_ll / n,
        "model_brier": model_brier / n,
        "market_brier": market_brier / n,
        "model_hit": model_hit / n,
        "market_hit": market_hit / n,
        "blend_n": blend_n,
        "blend_logloss": blend_ll / blend_n if blend_n else None,
        "blend_base_logloss": blend_base_ll / blend_n if blend_n else None,
        "blend_brier": blend_brier / blend_n if blend_n else None,
        "blend_base_brier": blend_base_brier / blend_n if blend_n else None,
        "blend_hit": blend_hit / blend_n if blend_n else None,
        "blend_base_hit": blend_base_hit / blend_n if blend_n else None,
    }
