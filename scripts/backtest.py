"""Walk-forward backtest of the prediction engine on the most recent results.

For every league, fetches the same match pool the app trains on, then
predicts the last N completed fixtures using only results dated before each
fixture — mirroring how the deployed model refits as results land.

Usage:
    .venv\\Scripts\\python.exe scripts\\backtest.py [--last 150] [--leagues PL,BL1]

Reports per-league and overall:
    hit rate (argmax outcome), log-loss, Brier score,
    mean predicted vs actual rates (home win / draw / away win).
"""

import argparse
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.data.league_service import LeagueService
from src.data.prediction_engine import PredictionEngine
from src.data.utils import compute_table

logger = logging.getLogger(__name__)


def _parse_date(m):
    try:
        return datetime.fromisoformat(m.get("date") or "")
    except (ValueError, TypeError):
        return None


def _contexts_from_table(matches, competition):
    """Build engine contexts from results known so far (same-competition table)."""
    table = compute_table([m for m in matches if m.get("competition") == competition])
    size = len(table)
    ctx = {}
    for row in table:
        played = row.get("played") or 0
        ctx[row["team"]] = {
            "position": row.get("position"),
            "league_size": size,
            "ppg": (row.get("points", 0) / played) if played else None,
            "competition": competition,
        }
    return ctx


def evaluate_league(code, last_n):
    """Returns (per-match eval rows, service)."""
    svc = LeagueService(code)
    pool = list(svc.matches) + list(svc._historical_matches)
    for data in svc._related.values():
        pool.extend(data["matches"])
    pool = [
        m
        for m in pool
        if m.get("status") == "full_time" and m.get("home_score") is not None
    ]
    pool.sort(key=lambda m: m.get("date") or "")

    own = [m for m in pool if m.get("competition") == code]
    test = own[-last_n:]
    if not test:
        return [], svc

    rows = []
    # Refit once per calendar day of test matches — predictions within a
    # matchday share the same information set.
    by_day = defaultdict(list)
    for m in test:
        by_day[(m.get("date") or "")[:10]].append(m)

    for day in sorted(by_day):
        train = [m for m in pool if (m.get("date") or "")[:10] < day]
        if not train:
            continue
        eng = PredictionEngine()
        eng.fit(train, initial_elo=svc._elo_seed(), primary_competition=code)
        ctx = _contexts_from_table(train, code)
        for m in by_day[day]:
            home, away = m["home_team"], m["away_team"]
            pred = eng.predict(
                home,
                away,
                competition=code,
                contexts={
                    eng.resolve_name(home): ctx.get(eng.resolve_name(home)),
                    eng.resolve_name(away): ctx.get(eng.resolve_name(away)),
                },
            )
            probs = pred["outcome_probabilities"]
            hs, aws = m["home_score"], m["away_score"]
            actual = "home_win" if hs > aws else ("draw" if hs == aws else "away_win")
            predicted = max(probs, key=probs.get)
            p_actual = max(probs.get(actual, 0.0), 1e-9)
            vec = [probs["home_win"], probs["draw"], probs["away_win"]]
            act = [
                1.0 if actual == "home_win" else 0.0,
                1.0 if actual == "draw" else 0.0,
                1.0 if actual == "away_win" else 0.0,
            ]
            rows.append(
                {
                    "league": code,
                    "date": m.get("date"),
                    "match": f"{home} {hs}-{aws} {away}",
                    "actual": actual,
                    "predicted": predicted,
                    "hit": predicted == actual,
                    "log_loss": -math.log(p_actual),
                    "brier": sum((p - a) ** 2 for p, a in zip(vec, act)),
                    "p_home": vec[0],
                    "p_draw": vec[1],
                    "p_away": vec[2],
                }
            )
    return rows, svc


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    hits = sum(r["hit"] for r in rows)
    ll = sum(r["log_loss"] for r in rows) / n
    brier = sum(r["brier"] for r in rows) / n
    p_home = sum(r["p_home"] for r in rows) / n
    p_draw = sum(r["p_draw"] for r in rows) / n
    p_away = sum(r["p_away"] for r in rows) / n
    a_home = sum(1 for r in rows if r["actual"] == "home_win") / n
    a_draw = sum(1 for r in rows if r["actual"] == "draw") / n
    a_away = sum(1 for r in rows if r["actual"] == "away_win") / n
    return {
        "n": n,
        "hit_rate": hits / n,
        "log_loss": ll,
        "brier": brier,
        "pred_home/draw/away": (p_home, p_draw, p_away),
        "actual_home/draw/away": (a_home, a_draw, a_away),
    }


def main():
    ap = argparse.ArgumentParser()
    # ~150 per league (1,000+ pooled) — smaller holdouts are too noisy
    # to tell parameter changes apart.
    ap.add_argument("--last", type=int, default=150, help="recent results per league to score")
    ap.add_argument("--leagues", type=str, default="", help="comma-separated league codes")
    ap.add_argument("--splits", type=int, default=1,
                    help="split the holdout chronologically into N blocks "
                         "and report each — reveals variance across cutoffs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    svc_probe = LeagueService("PL")
    codes = (
        args.leagues.split(",")
        if args.leagues
        else list(svc_probe._leagues.keys())
    )

    all_rows = []
    per_league = {}
    for code in codes:
        try:
            rows, _ = evaluate_league(code, args.last)
        except Exception as exc:
            print(f"{code:8s} FAILED: {exc}")
            continue
        per_league[code] = rows
        all_rows.extend(rows)

    print(f"\n{'league':8s} {'n':>3s} {'hit%':>6s} {'logloss':>8s} {'brier':>6s} "
          f"{'pred H/D/A':>22s} {'actual H/D/A':>22s}")
    for code, rows in per_league.items():
        s = summarize(rows)
        if not s:
            print(f"{code:8s}   0  (no completed recent matches)")
            continue
        ph = " / ".join(f"{v:.2f}" for v in s["pred_home/draw/away"])
        ah = " / ".join(f"{v:.2f}" for v in s["actual_home/draw/away"])
        print(f"{code:8s} {s['n']:>3d} {s['hit_rate']*100:>5.1f}% {s['log_loss']:>8.3f} "
              f"{s['brier']:>6.3f} {ph:>22s} {ah:>22s}")

    s = summarize(all_rows)
    if s:
        ph = " / ".join(f"{v:.2f}" for v in s["pred_home/draw/away"])
        ah = " / ".join(f"{v:.2f}" for v in s["actual_home/draw/away"])
        print(f"\n{'ALL':8s} {s['n']:>3d} {s['hit_rate']*100:>5.1f}% {s['log_loss']:>8.3f} "
              f"{s['brier']:>6.3f} {ph:>22s} {ah:>22s}")

    if args.splits > 1 and all_rows:
        # Chronological blocks of the pooled holdout — the same knob can
        # "win" on one block and lose on another, so check the spread.
        ordered = sorted(all_rows, key=lambda r: r["date"])
        k = max(1, (len(ordered) + args.splits - 1) // args.splits)
        blocks = [ordered[i * k:(i + 1) * k] for i in range(args.splits)]
        print(f"\n--- {args.splits} chronological blocks ---")
        for i, block in enumerate(blocks, 1):
            s = summarize(block)
            if not s:
                continue
            print(f"block {i}  n={s['n']:>3d}  hit={s['hit_rate']*100:5.1f}%  "
                  f"ll={s['log_loss']:.4f}  brier={s['brier']:.4f}")

    if args.verbose:
        print("\n--- misses and near-misses ---")
        for r in sorted(all_rows, key=lambda r: -r["log_loss"])[:25]:
            print(
                f"{r['league']:8s} {r['date'][:10]}  {r['match']:<55s} "
                f"actual={r['actual']:<8s} pred={r['predicted']:<8s} ll={r['log_loss']:.2f}"
            )


if __name__ == "__main__":
    main()
