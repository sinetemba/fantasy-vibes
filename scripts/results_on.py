"""List latest results — every match played on a given date.

Usage:
    .venv\\Scripts\\python.exe scripts\\results_on.py --date YYYY-MM-DD [--leagues PL,BL1,...]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.data.league_service import LeagueService
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
    ap.add_argument("--date", type=str, required=True)
    ap.add_argument("--leagues", type=str, default="")
    args = ap.parse_args()

    bust_live_cache(3 * 3600)

    probe = LeagueService("PL")
    codes = args.leagues.split(",") if args.leagues else list(probe._leagues.keys())

    for code in codes:
        try:
            svc = probe if code == "PL" else LeagueService(code)
        except Exception as exc:
            _log(f"{code:8s} FAILED to load: {exc}")
            continue
        day = [m for m in svc.get_matches() if (m.get("date") or "")[:10] == args.date]
        if not day:
            continue
        _log(f"\n== {code} {svc.get_current_name()} ==")
        for m in sorted(day, key=lambda x: x.get("date") or ""):
            hs, as_ = m.get("home_score"), m.get("away_score")
            score = f"{hs}-{as_}" if hs is not None and as_ is not None else " - "
            _log(
                f"  {m.get('status','?'):10s} {m.get('home_team','?'):30s} "
                f"{score:5s} {m.get('away_team','?'):30s} {m.get('date')}"
            )


if __name__ == "__main__":
    main()
