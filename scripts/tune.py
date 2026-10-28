"""Coordinate-style parameter search over the prediction engine constants.

Loads every league's match pool once, then for each candidate value of each
tunable constant re-runs the same walk-forward evaluation used by
backtest.py and ranks by mean log-loss (with Brier/hit-rate for context).

Usage:
    .venv\\Scripts\\python.exe scripts\\tune.py [--last 150] [--leagues PL,BL1,...]
"""

import argparse
import itertools
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import src.data.prediction_engine as pe
from src.data.league_service import LeagueService
from src.data.utils import compute_table

logger = logging.getLogger(__name__)

# Candidate values per knob — baseline (current file defaults) included.
GRID = {
    "RECENCY_HALF_LIFE_DAYS": [45.0, 60.0, 90.0, 120.0, 180.0],
    "BLEND_POISSON": [0.35, 0.5, 0.65],
    "DRAW_FLOOR": [0.10, 0.14, 0.18],
    "DRAW_ELO_SCALE": [900.0, 1200.0, 1800.0],
    "DRAW_RATE_MAX": [0.30, 0.35, 0.40],
    "ELO_K": [20.0, 30.0, 40.0],
    "ELO_HOME_FIELD": [50.0, 70.0, 100.0],
    "ATTACK_SHRINK_GAMES": [4.0, 8.0, 12.0],
    "FORM_WEIGHT": [0.1, 0.2, 0.3],
}


def load_league(code, last_n):
    svc = LeagueService(code)
    pool = list(svc.matches) + list(svc._historical_matches)
    for data in svc._related.values():
        pool.extend(data["matches"])
    pool = [
        m for m in pool
        if m.get("status") == "full_time" and m.get("home_score") is not None
    ]
    pool.sort(key=lambda m: m.get("date") or "")
    own = [m for m in pool if m.get("competition") == code]
    test = own[-last_n:]
    by_day = defaultdict(list)
    for m in test:
        by_day[(m.get("date") or "")[:10]].append(m)
    return {
        "code": code,
        "svc": svc,
        "pool": pool,
        "days": sorted(by_day.items()),
    }


def contexts_for(train, competition):
    table = compute_table([m for m in train if m.get("competition") == competition])
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


def score_league(lg):
    """Walk-forward evaluation under the current module constants."""
    rows = []
    code = lg["code"]
    seed = lg["svc"]._elo_seed()
    for day, matches in lg["days"]:
        train = [m for m in lg["pool"] if (m.get("date") or "")[:10] < day]
        if not train:
            continue
        eng = pe.PredictionEngine()
        eng.fit(train, initial_elo=seed, primary_competition=code)
        ctx = contexts_for(train, code)
        for m in matches:
            home, away = m["home_team"], m["away_team"]
            pred = eng.predict(
                home, away, competition=code,
                contexts={
                    eng.resolve_name(home): ctx.get(eng.resolve_name(home)),
                    eng.resolve_name(away): ctx.get(eng.resolve_name(away)),
                },
            )
            probs = pred["outcome_probabilities"]
            hs, aws = m["home_score"], m["away_score"]
            actual = "home_win" if hs > aws else ("draw" if hs == aws else "away_win")
            p = max(probs.get(actual, 0.0), 1e-9)
            vec = [probs["home_win"], probs["draw"], probs["away_win"]]
            act = [1.0 if actual == k else 0.0 for k in ("home_win", "draw", "away_win")]
            rows.append({
                "hit": max(probs, key=probs.get) == actual,
                "log_loss": -math.log(p),
                "brier": sum((a - b) ** 2 for a, b in zip(vec, act)),
                "p_draw": vec[1],
                "is_draw": actual == "draw",
            })
    return rows


def evaluate(leagues):
    tot = {"n": 0, "hits": 0, "ll": 0.0, "brier": 0.0, "p_draw": 0.0, "n_draw": 0}
    for lg in leagues:
        for r in score_league(lg):
            tot["n"] += 1
            tot["hits"] += r["hit"]
            tot["ll"] += r["log_loss"]
            tot["brier"] += r["brier"]
            tot["p_draw"] += r["p_draw"]
            tot["n_draw"] += r["is_draw"]
    n = tot["n"]
    if not n:
        return None
    return {
        "n": n,
        "hit": tot["hits"] / n,
        "ll": tot["ll"] / n,
        "brier": tot["brier"] / n,
        "p_draw": tot["p_draw"] / n,
        "a_draw": tot["n_draw"] / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=150)  # see backtest.py
    ap.add_argument("--leagues", type=str, default="")
    ap.add_argument("--full", action="store_true",
                    help="Also try pairwise combos of the per-knob winners")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    probe = LeagueService("PL")
    codes = args.leagues.split(",") if args.leagues else list(probe._leagues.keys())

    leagues = []
    for code in codes:
        try:
            lg = load_league(code, args.last)
            n = sum(len(v) for _, v in lg["days"])
            print(f"loaded {code}: pool={len(lg['pool'])} test={n}", flush=True)
            if n >= 3:
                leagues.append(lg)
        except Exception as exc:
            print(f"{code}: load failed: {exc}", flush=True)

    base = {k: getattr(pe, k) for k in GRID}
    results = []

    def run(tag):
        s = evaluate(leagues)
        if s:
            results.append((s["ll"], tag, dict(
                n=s["n"], hit=s["hit"], brier=s["brier"],
                p_draw=s["p_draw"], a_draw=s["a_draw"])))
            print(f"{tag:55s} ll={s['ll']:.4f} brier={s['brier']:.4f} "
                  f"hit={s['hit']*100:.1f}% pdraw={s['p_draw']:.3f} (actual {s['a_draw']:.3f})",
                  flush=True)

    run("baseline")
    winners = {}
    for knob, values in GRID.items():
        for v in values:
            if v == base[knob]:
                continue
            setattr(pe, knob, v)
            run(f"{knob}={v}")
        setattr(pe, knob, base[knob])
        # Best value for this knob (excluding baseline tag).
        knob_results = [r for r in results if r[1].startswith(f"{knob}=")]
        if knob_results:
            best_tag = min(knob_results)[1]
            winners[knob] = float(best_tag.split("=")[1])
            setattr(pe, knob, winners[knob])
            print(f"-> adopt {knob}={winners[knob]}", flush=True)

    run("coordinate winners")

    if args.full:
        knobs = list(winners)
        for a, b in itertools.combinations(knobs, 2):
            saved = {a: getattr(pe, a), b: getattr(pe, b)}
            for va, vb in itertools.product(GRID[a], GRID[b]):
                setattr(pe, a, va)
                setattr(pe, b, vb)
                run(f"{a}={va} + {b}={vb}")
            for k in (a, b):
                setattr(pe, k, saved[k])
        for k, v in winners.items():
            setattr(pe, k, v)
        run("winners recheck")

    print("\n=== adopted values ===")
    for k, v in winners.items():
        print(f"{k} = {v}   (was {base[k]})")
    print("\n=== top 10 configs ===")
    for ll, tag, s in sorted(results)[:10]:
        print(f"{ll:.4f}  {tag}")


if __name__ == "__main__":
    main()
