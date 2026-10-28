"""Prediction log: records each fixture's last pre-kickoff prediction and
scores it once the result lands, so the app can report its own recent
accuracy instead of asking users to take the model on trust."""

import json
import logging
import math
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .prediction_engine import fixtures_equivalent
from .utils import sast_now, to_sast

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = PROJECT_ROOT / "data" / "cache" / "prediction_log.json"
# Settled and abandoned (never-played) entries are dropped after this long.
RETENTION_DAYS = 120

_lock = threading.Lock()
_entries: Optional[Dict[str, Dict[str, Any]]] = None


def kickoff_sast(raw: Optional[str]) -> Optional[datetime]:
    """Kickoff as naive SAST — feeds store naive SAST, a few carry offsets."""
    try:
        dt = datetime.fromisoformat(raw or "")
    except (ValueError, TypeError):
        return None
    return to_sast(dt) if dt.tzinfo else dt


def _load() -> Dict[str, Dict[str, Any]]:
    global _entries
    if _entries is None:
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                _entries = json.load(f).get("entries", {})
        except FileNotFoundError:
            _entries = {}
        except Exception as exc:
            logger.warning(f"Prediction log unreadable, starting fresh: {exc}")
            _entries = {}
    return _entries


def _save(entries: Dict[str, Dict[str, Any]]) -> None:
    cutoff = sast_now() - timedelta(days=RETENTION_DAYS)
    for key in [k for k, e in entries.items() if (kickoff_sast(e["date"]) or cutoff) <= cutoff]:
        del entries[key]
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
            entries[key] = {
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


def summary(days: int = 30) -> Dict[str, Any]:
    """Accuracy of settled predictions whose kickoff fell in the last
    `days` days: hit rate (likeliest outcome), Brier score, log-loss and
    the hit rate of always picking the home side, for context."""
    cutoff = sast_now() - timedelta(days=days)
    with _lock:
        settled = [
            dict(e) for e in _load().values()
            if e.get("result") and (kickoff_sast(e["date"]) or cutoff) > cutoff
        ]
    n = len(settled)
    if not n:
        return {"n": 0, "days": days}
    keys = {"home_win": "p_home", "draw": "p_draw", "away_win": "p_away"}
    hits = brier = log_loss = 0.0
    per_league: Dict[str, List[int]] = {}
    for e in settled:
        probs = {outcome: float(e[k]) for outcome, k in keys.items()}
        hit = max(probs, key=probs.get) == e["result"]
        hits += hit
        brier += sum((p - (o == e["result"])) ** 2 for o, p in probs.items())
        log_loss += -math.log(max(probs[e["result"]], 1e-9))
        tally = per_league.setdefault(e["league"], [0, 0])
        tally[0] += hit
        tally[1] += 1
    return {
        "n": n,
        "days": days,
        "hit_rate": hits / n,
        "brier": brier / n,
        "log_loss": log_loss / n,
        "home_baseline": sum(e["result"] == "home_win" for e in settled) / n,
        "per_league": {lg: (h / c, c) for lg, (h, c) in sorted(per_league.items())},
    }
