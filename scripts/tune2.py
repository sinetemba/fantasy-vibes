"""Focused config comparison — named parameter sets over the cached pools."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import src.data.prediction_engine as pe
from tune import load_league, evaluate

CONFIGS = {
    "baseline (180d)": {},
    "recency 120": {"RECENCY_HALF_LIFE_DAYS": 120.0},
    "recency 240": {"RECENCY_HALF_LIFE_DAYS": 240.0},
    "recency 300": {"RECENCY_HALF_LIFE_DAYS": 300.0},
    "blend .65": {"BLEND_POISSON": 0.65},
    "blend .80": {"BLEND_POISSON": 0.8},
    "form .3": {"FORM_WEIGHT": 0.3},
    "form .45": {"FORM_WEIGHT": 0.45},
    "shrink 4": {"ATTACK_SHRINK_GAMES": 4.0},
    "shrink 6": {"ATTACK_SHRINK_GAMES": 6.0},
    "elo hf 50": {"ELO_HOME_FIELD": 50.0},
    "elo k20": {"ELO_K": 20.0},
    "draw loose": {"DRAW_FLOOR": 0.14, "DRAW_ELO_SCALE": 1800.0, "DRAW_RATE_MAX": 0.40},
    "combo A": {
        "RECENCY_HALF_LIFE_DAYS": 120.0, "BLEND_POISSON": 0.65,
        "FORM_WEIGHT": 0.3, "ATTACK_SHRINK_GAMES": 4.0,
    },
    "combo B": {
        "BLEND_POISSON": 0.65, "FORM_WEIGHT": 0.3, "ATTACK_SHRINK_GAMES": 4.0,
        "DRAW_FLOOR": 0.14, "DRAW_ELO_SCALE": 1800.0, "DRAW_RATE_MAX": 0.40,
        "ELO_HOME_FIELD": 50.0, "ELO_K": 20.0,
    },
    "combo C": {
        "RECENCY_HALF_LIFE_DAYS": 240.0, "BLEND_POISSON": 0.65,
        "FORM_WEIGHT": 0.3, "ATTACK_SHRINK_GAMES": 6.0,
        "DRAW_FLOOR": 0.14, "DRAW_ELO_SCALE": 1800.0, "DRAW_RATE_MAX": 0.40,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--leagues", type=str,
                    default="PL,BL1,SA,PD,FL1,PSL,CL,EL,INTF,UNL,WCQUEFA,GCQ")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    leagues = []
    for code in args.leagues.split(","):
        try:
            lg = load_league(code, args.last)
            n = sum(len(v) for _, v in lg["days"])
            print(f"loaded {code}: pool={len(lg['pool'])} test={n}", flush=True)
            if n >= 3:
                leagues.append(lg)
        except Exception as exc:
            print(f"{code}: load failed: {exc}", flush=True)

    baseline_vals = {k: getattr(pe, k) for cfg in CONFIGS.values() for k in cfg}
    for name, cfg in CONFIGS.items():
        for k in baseline_vals:
            setattr(pe, k, baseline_vals[k])
        for k, v in cfg.items():
            setattr(pe, k, v)
        s = evaluate(leagues)
        print(f"{name:20s} ll={s['ll']:.4f} brier={s['brier']:.4f} "
              f"hit={s['hit']*100:.1f}% pdraw={s['p_draw']:.3f} (actual {s['a_draw']:.3f})",
              flush=True)


if __name__ == "__main__":
    main()
