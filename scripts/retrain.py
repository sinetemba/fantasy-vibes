"""Retrain every league's prediction model on the latest results.

Fetches current data via LeagueService (same cascade the app uses), forces
a refit, persists data/cache/<CODE>_model.json, and refreshes
data/cache/last_retrain_report.json.

Usage:
    .venv\\Scripts\\python.exe scripts\\retrain.py [--leagues PL,BL1,...]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.data.league_service import LeagueService, MODEL_CACHE_DIR
from src.data.utils import bust_live_cache


def _log(msg):
    try:
        print(msg, flush=True)
    except OSError:
        try:
            sys.stderr.write(msg + "\n")
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", type=str, default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Collapse HTTP cache TTLs for the whole run so training data is the
    # freshest available rather than up-to-30-minute-old cached results.
    bust_live_cache(3 * 3600)

    probe = LeagueService("PL")
    codes = args.leagues.split(",") if args.leagues else list(probe._leagues.keys())

    report = {"retrained_at": datetime.now().isoformat(), "leagues": []}
    for code in codes:
        try:
            svc = probe if code == "PL" else LeagueService(code)
            status = svc.train()
        except Exception as exc:
            _log(f"{code:8s} FAILED: {exc}")
            report["leagues"].append({"league": code, "error": str(exc)})
            continue

        ft = [
            m for m in svc.matches
            if m.get("status") == "full_time" and m.get("home_score") is not None
        ]
        ft.sort(key=lambda m: m.get("date") or "")
        latest = ft[-1] if ft else {}
        entry = {
            "league": code,
            "name": svc.get_current_name(),
            "status": status,
            "full_time_matches": len(ft),
            "recent_result_date": latest.get("date"),
            "recent_result": (
                f"{latest.get('home_team')} {latest.get('home_score')}-"
                f"{latest.get('away_score')} {latest.get('away_team')}"
                if latest else None
            ),
        }
        report["leagues"].append(entry)
        _log(
            f"{code:8s} fitted={status['fitted']} teams={status['teams_in_model']:>3} "
            f"train_matches={status['matches_used']:>4} latest={entry['recent_result']}"
        )

    out = MODEL_CACHE_DIR / "last_retrain_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _log(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
