"""Prediction log: records each fixture's last pre-kickoff prediction and
scores it once the result lands, so the app can report its own recent
accuracy instead of asking users to take the model on trust."""

import json
import logging
import math
import os
import shutil
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .prediction_engine import fixtures_equivalent
from .utils import sast_now, to_sast

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Lives in data/, not data/cache/: the log is the app's only record of its
# own track record and must not sit next to disposable HTTP cache files.
LOG_PATH = PROJECT_ROOT / "data" / "prediction_log.json"
# Daily snapshots kept under data/backups/ so a bad write or accidental
# delete costs at most a day of entries.
BACKUP_KEEP = 14
# Settled and abandoned (never-played) entries are dropped after this long
# — a full season, so the track record and history cover one.
RETENTION_DAYS = 365

# Goal lines whose over-probability is logged (keys of the prediction's
# over_under markets) and the entry field each is stored under.
GOAL_LINE_FIELDS = {"1.5": "p_over15", "2.5": "p_over25", "3.5": "p_over35"}

_lock = threading.Lock()
_entries: Optional[Dict[str, Dict[str, Any]]] = None


def kickoff_sast(raw: Optional[str]) -> Optional[datetime]:
    """Kickoff as naive SAST — feeds store naive SAST, a few carry offsets."""
    try:
        dt = datetime.fromisoformat(raw or "")
    except (ValueError, TypeError):
        return None
    return to_sast(dt) if dt.tzinfo else dt


def _legacy_path() -> Path:
    """Where the log lived before it moved out of the cache directory."""
    return LOG_PATH.parent / "cache" / LOG_PATH.name


def _backup_dir() -> Path:
    return LOG_PATH.parent / "backups"


def _migrate_legacy() -> None:
    legacy = _legacy_path()
    if LOG_PATH.exists() or not legacy.exists():
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        os.replace(legacy, LOG_PATH)
        logger.info(f"Moved prediction log {legacy} -> {LOG_PATH}")
    except Exception as exc:
        logger.warning(f"Could not move legacy prediction log: {exc}")


def _load() -> Dict[str, Dict[str, Any]]:
    global _entries
    if _entries is None:
        _migrate_legacy()
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                _entries = json.load(f).get("entries", {})
        except FileNotFoundError:
            _entries = {}
        except Exception as exc:
            # Set the unreadable file aside rather than letting the next
            # save overwrite it — it may be recoverable by hand.
            aside = LOG_PATH.with_name(
                f"{LOG_PATH.stem}.corrupt-{int(time.time())}{LOG_PATH.suffix}"
            )
            try:
                os.replace(LOG_PATH, aside)
            except Exception:
                aside = None
            logger.error(
                f"Prediction log unreadable ({exc}); starting fresh"
                + (f", original kept at {aside}" if aside else "")
                + f". Daily backups are in {_backup_dir()}."
            )
            _entries = {}
    return _entries


def _backup_daily() -> None:
    """Snapshot the on-disk log once per day, keeping the newest BACKUP_KEEP."""
    if not LOG_PATH.exists():
        return
    folder = _backup_dir()
    target = folder / f"{LOG_PATH.stem}-{sast_now():%Y-%m-%d}{LOG_PATH.suffix}"
    if target.exists():
        return
    try:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LOG_PATH, target)
        snapshots = sorted(folder.glob(f"{LOG_PATH.stem}-*{LOG_PATH.suffix}"))
        for old in snapshots[:-BACKUP_KEEP]:
            old.unlink()
    except Exception as exc:
        logger.warning(f"Could not back up prediction log: {exc}")


def _save(entries: Dict[str, Dict[str, Any]]) -> None:
    cutoff = sast_now() - timedelta(days=RETENTION_DAYS)
    for key in [k for k, e in entries.items() if (kickoff_sast(e["date"]) or cutoff) <= cutoff]:
        del entries[key]
    _backup_daily()
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LOG_PATH.with_name(LOG_PATH.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "entries": entries}, f)
        os.replace(tmp, LOG_PATH)
    except Exception as exc:
        logger.warning(f"Could not save prediction log: {exc}")


def _near(entry: Dict[str, Any], m: Dict[str, Any]) -> bool:
    """Cheap date pre-filter (±1 day) before the costlier fixture match."""
    a, b = kickoff_sast(entry["date"]), kickoff_sast(m.get("date"))
    return bool(a and b and abs((a - b).total_seconds()) <= 36 * 3600)


def record(league: str, pairs: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
    """Log (fixture, prediction) pairs for fixtures that haven't kicked off.
    A fixture already logged — under any feed's id — is overwritten, so the
    log keeps the latest prediction made before kickoff."""
    now = sast_now()
    with _lock:
        entries = _load()
        changed = False
        for m, pred in pairs:
            mid = m.get("match_id")
            kickoff = kickoff_sast(m.get("date"))
            if not mid or m.get("status") != "scheduled" or not kickoff or kickoff <= now:
                continue
            key = f"{league}|{mid}"
            if key not in entries:
                # The same fixture may be logged from another feed's row.
                key = next(
                    (
                        k for k, e in entries.items()
                        if e["league"] == league and e.get("result") is None
                        and _near(e, m) and fixtures_equivalent(e, m)
                    ),
                    key,
                )
            probs = pred.get("outcome_probabilities") or {}
            entry = {
                "league": league,
                "match_id": mid,
                "home_team": m.get("home_team"),
                "away_team": m.get("away_team"),
                "date": m.get("date"),
                "predicted_at": now.isoformat(timespec="seconds"),
                "model": pred.get("model"),
                "p_home": probs.get("home_win", 0.0),
                "p_draw": probs.get("draw", 0.0),
                "p_away": probs.get("away_win", 0.0),
                "result": None,
            }
            # Goals markets and confidence, so they're scored like outcomes.
            over_under = pred.get("over_under") or {}
            for line, field in GOAL_LINE_FIELDS.items():
                if (over_under.get(line) or {}).get("over") is not None:
                    entry[field] = over_under[line]["over"]
            if (pred.get("btts") or {}).get("yes") is not None:
                entry["p_btts"] = pred["btts"]["yes"]
            if pred.get("confidence"):
                entry["confidence"] = pred["confidence"]
            if pred.get("unrated_teams"):
                entry["unrated"] = list(pred["unrated_teams"])
            score = (pred.get("predicted_score") or {}).get("score")
            if score:
                entry["predicted_score"] = score
            # Uncalibrated model probs (before temperature scaling and the
            # market blend) — the only input temperature() fits on.
            raw_probs = pred.get("raw_probabilities")
            if raw_probs:
                entry["r_home"] = raw_probs.get("home_win")
                entry["r_draw"] = raw_probs.get("draw")
                entry["r_away"] = raw_probs.get("away_win")
            # Pre-blend model probs and the market probs blended in — p_*
            # is what users saw, but scoring the blend against the market
            # it contains is circular. Keeping the components lets the
            # benchmark honestly split model vs blend vs market.
            model_probs = pred.get("model_probabilities")
            if model_probs:
                entry["m_home"] = model_probs.get("home_win")
                entry["m_draw"] = model_probs.get("draw")
                entry["m_away"] = model_probs.get("away_win")
            market_probs = pred.get("market_probabilities")
            if market_probs:
                entry["k_home"] = market_probs.get("home_win")
                entry["k_draw"] = market_probs.get("draw")
                entry["k_away"] = market_probs.get("away_win")
            entries[key] = entry
            changed = True
        if changed:
            _save(entries)


def settle(league: str, matches: List[Dict[str, Any]]) -> None:
    """Attach full-time results to logged predictions for this league,
    matching by id first and by fixture equivalence across feeds second."""
    finished = [
        m for m in matches
        if m.get("status") == "full_time"
        and m.get("home_score") is not None
        and m.get("away_score") is not None
    ]
    if not finished:
        return
    by_id = {m.get("match_id"): m for m in finished if m.get("match_id")}
    with _lock:
        entries = _load()
        changed = False
        for e in entries.values():
            if e["league"] != league or e.get("result") is not None:
                continue
            m = by_id.get(e["match_id"])
            if m is None or not fixtures_equivalent(e, m):
                m = next(
                    (f for f in finished if _near(e, f) and fixtures_equivalent(e, f)),
                    None,
                )
            if m is None:
                continue
            hs, aws = m["home_score"], m["away_score"]
            e["home_score"], e["away_score"] = hs, aws
            e["result"] = "home_win" if hs > aws else ("draw" if hs == aws else "away_win")
            changed = True
        if changed:
            _save(entries)


# Fit window for outcome-probability temperature scaling.
_RESULT_IDX = {"home_win": 0, "draw": 1, "away_win": 2}


def _settled_entries(days: int) -> List[Dict[str, Any]]:
    """Settled log entries whose kickoff fell within the last `days` days."""
    cutoff = sast_now() - timedelta(days=days)
    with _lock:
        return [
            dict(e)
            for e in _load().values()
            if e.get("result") in _RESULT_IDX
            and (kickoff_sast(e["date"]) or cutoff) > cutoff
        ]


def _fit_temperature(rows: List[Tuple[float, float, float, str]]) -> float:
    """Golden-section search for the temperature minimising NLL of the
    (p_home, p_draw, p_away, result) rows."""

    def nll(T: float) -> float:
        loss = 0.0
        for h, d, a, result in rows:
            ps = [max(float(p), 1e-12) ** (1.0 / T) for p in (h, d, a)]
            total = sum(ps)
            loss -= math.log(max(ps[_RESULT_IDX[result]] / total, 1e-12))
        return loss

    lo, hi = 0.5, 3.0
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = nll(x1), nll(x2)
    for _ in range(60):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = nll(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = nll(x2)
    return (lo + hi) / 2.0


def temperature(league: str, days: int = 120, min_n: int = 120) -> Optional[float]:
    """Temperature that rescales this league's logged probabilities to
    match observed outcomes: T > 1 means the model is overconfident and
    its predictions should be flattened toward uniform.

    A league-specific fit requires `min_n` settled entries; below that the
    pooled fit across every settled league is used instead, so low-volume
    competitions still benefit from calibration. None when neither pool
    reaches `min_n` or when the fitted T ~ 1 (nothing to correct).

    Fits only on the uncalibrated r_* probabilities. p_* has already been
    scaled by the T in force when it was logged (and possibly blended with
    the market), so fitting on it would find T ~ 1 once calibration works,
    switch calibration off, and oscillate. Entries without r_* (logged
    before it was recorded) are skipped."""
    rows = [
        (e["r_home"], e["r_draw"], e["r_away"], e["result"], e["league"])
        for e in _settled_entries(days)
        if e.get("r_home") is not None
        and e.get("r_draw") is not None
        and e.get("r_away") is not None
    ]
    league_rows = [r[:4] for r in rows if r[4] == league]
    pooled_rows = [r[:4] for r in rows]
    if len(league_rows) >= min_n:
        T = _fit_temperature(league_rows)
    elif len(pooled_rows) >= min_n:
        T = _fit_temperature(pooled_rows)
    else:
        return None
    return None if abs(T - 1.0) < 0.05 else round(T, 3)


_OUTCOME_KEYS = {"home_win": "p_home", "draw": "p_draw", "away_win": "p_away"}


def wilson_interval(hits: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a hit rate — honest error bars that
    stay inside [0, 1] even for small samples."""
    if n <= 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _outcome_hit(e: Dict[str, Any]) -> bool:
    probs = {outcome: float(e[k]) for outcome, k in _OUTCOME_KEYS.items()}
    return max(probs, key=probs.get) == e["result"]


def _binary_market(rows: List[Tuple[float, bool]]) -> Optional[Dict[str, Any]]:
    """Hit rate (call the side above 50%), Brier score and how often the
    event happened, for a yes/no market's (p_yes, happened) rows."""
    n = len(rows)
    if not n:
        return None
    hits = sum((p >= 0.5) == happened for p, happened in rows)
    return {
        "n": n,
        "hit_rate": hits / n,
        "hit_rate_ci": wilson_interval(hits, n),
        "brier": sum((p - happened) ** 2 for p, happened in rows) / n,
        "base_rate": sum(happened for _, happened in rows) / n,
    }


def summary(days: int = 30) -> Dict[str, Any]:
    """Accuracy of settled predictions whose kickoff fell in the last
    `days` days: hit rate (likeliest outcome) with a 95% interval, Brier
    score, log-loss, the hit rate of always picking the home side for
    context, hit rate per confidence level, and the goals markets."""
    settled = _settled_entries(days)
    n = len(settled)
    if not n:
        return {"n": 0, "days": days}
    hits = brier = log_loss = 0.0
    per_league: Dict[str, List[int]] = {}
    per_conf: Dict[str, List[int]] = {}
    for e in settled:
        probs = {outcome: float(e[k]) for outcome, k in _OUTCOME_KEYS.items()}
        hit = _outcome_hit(e)
        hits += hit
        brier += sum((p - (o == e["result"])) ** 2 for o, p in probs.items())
        log_loss += -math.log(max(probs[e["result"]], 1e-9))
        tally = per_league.setdefault(e["league"], [0, 0])
        tally[0] += hit
        tally[1] += 1
        if e.get("confidence"):
            tally = per_conf.setdefault(e["confidence"], [0, 0])
            tally[0] += hit
            tally[1] += 1

    scored = [
        e for e in settled
        if e.get("home_score") is not None and e.get("away_score") is not None
    ]
    goals: Dict[str, Any] = {}
    for line, field in GOAL_LINE_FIELDS.items():
        market = _binary_market([
            (float(e[field]), e["home_score"] + e["away_score"] > float(line))
            for e in scored if e.get(field) is not None
        ])
        if market:
            goals[f"over_{line}"] = market
    btts = _binary_market([
        (float(e["p_btts"]), e["home_score"] > 0 and e["away_score"] > 0)
        for e in scored if e.get("p_btts") is not None
    ])
    if btts:
        goals["btts"] = btts

    return {
        "n": n,
        "days": days,
        "hit_rate": hits / n,
        "hit_rate_ci": wilson_interval(hits, n),
        "brier": brier / n,
        "log_loss": log_loss / n,
        "home_baseline": sum(e["result"] == "home_win" for e in settled) / n,
        "per_league": {lg: (h / c, c) for lg, (h, c) in sorted(per_league.items())},
        "by_confidence": {c: (h / k, k) for c, (h, k) in per_conf.items()},
        "goals": goals,
    }


def unrated_teams(days: int = 30) -> List[Tuple[str, str, int]]:
    """(league, team, predictions) for teams logged with no rating at all
    — neither match history nor a prior — most frequent first. Each is a
    missing ranking entry or alias to fix."""
    cutoff = sast_now() - timedelta(days=days)
    counts: Counter = Counter()
    with _lock:
        for e in _load().values():
            if (kickoff_sast(e["date"]) or cutoff) > cutoff:
                for team in e.get("unrated") or []:
                    counts[(e["league"], team)] += 1
    return [(lg, team, n) for (lg, team), n in counts.most_common()]


def history(days: int = 30, league: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every logged prediction — settled, awaiting a result, or upcoming —
    whose kickoff is within the last `days` days or still ahead, newest
    kickoff first. Settled entries gain `correct` (likeliest outcome hit)."""
    cutoff = sast_now() - timedelta(days=days)
    with _lock:
        rows = [
            dict(e) for e in _load().values()
            if (league is None or e["league"] == league)
            and (kickoff_sast(e["date"]) or cutoff) > cutoff
        ]
    for e in rows:
        e["correct"] = _outcome_hit(e) if e.get("result") in _RESULT_IDX else None
    rows.sort(key=lambda e: kickoff_sast(e["date"]) or cutoff, reverse=True)
    return rows
