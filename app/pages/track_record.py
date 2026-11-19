"""Track record page: the model's live accuracy and a browsable history of
every prediction logged before kickoff — settled, pending and upcoming."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from app.ui import (
    CONFIDENCE_STYLES,
    apply_theme,
    display_disclaimer,
    display_header,
    league_name,
)
from src.data import LeagueService, odds, prediction_log

WINDOWS = {"Last 30 days": 30, "Last 90 days": 90, "Last 365 days": 365}
_PICK_LABELS = {"home_win": "Home", "draw": "Draw", "away_win": "Away"}


def render(service: LeagueService):
    apply_theme()
    display_header("Track Record", "How the predictions have actually done")

    window = st.segmented_control(
        "Period", list(WINDOWS), default="Last 30 days", key="track_window"
    ) or "Last 30 days"
    days = WINDOWS[window]

    report = prediction_log.summary(days=days)
    if not report["n"]:
        st.info(
            "Tracking is running: every fixture in the coming week gets a "
            "prediction logged before kickoff, and its accuracy appears here "
            "once results come in."
        )
    else:
        _render_outcomes(report)
        _render_confidence(report)
        _render_goals(report)
        _render_market_benchmark(days)

    _render_history(days, service)
    _render_unrated(days)

    st.markdown("---")
    display_disclaimer()


def _render_unrated(days: int) -> None:
    """Teams predicted with no rating at all — shown so gaps in the
    rankings/alias data are visible instead of silently weakening picks."""
    rows = prediction_log.unrated_teams(days=days)
    if not rows:
        return
    with st.expander(f"⚠️ Teams the model couldn't rate ({len(rows)})"):
        st.caption(
            "These teams had no match history and no ranking prior, so their "
            "predictions used a generic default rating and are less reliable."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"League": league_name(lg), "Team": team, "Predictions": n}
                    for lg, team, n in rows
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _ci_text(ci) -> str:
    lo, hi = ci
    return f"likely between {lo * 100:.0f}% and {hi * 100:.0f}%"


def _render_outcomes(report: Dict[str, Any]) -> None:
    st.markdown("### 📏 Match outcomes")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Likeliest outcome correct",
        _pct(report["hit_rate"]),
        help="How often the outcome the model rated most likely actually "
        "happened. The range is a 95% interval — small samples swing a lot.",
    )
    c2.metric(
        "Always picking the home side",
        _pct(report["home_baseline"]),
        help="The naive baseline the model should beat.",
    )
    c3.metric(
        "Brier score",
        f"{report['brier']:.3f}",
        help="Error of the full probability forecast — lower is better; "
        "0.667 is what blind one-third guesses score.",
    )
    leagues = " · ".join(
        f"{league_name(code)} {rate * 100:.0f}% ({n})"
        for code, (rate, n) in report["per_league"].items()
    )
    st.caption(
        f"{report['n']} fixtures settled in the last {report['days']} days, scored on "
        f"the last prediction made before kickoff. True hit rate "
        f"{_ci_text(report['hit_rate_ci'])}. By league: {leagues}"
    )


def _render_confidence(report: Dict[str, Any]) -> None:
    """Hit rate per confidence level — the labels only mean something if
    'high' really lands more often than 'low'."""
    by_conf = report.get("by_confidence") or {}
    if not by_conf:
        return
    st.markdown("#### 🎚️ By confidence level")
    levels = [lvl for lvl in ("high", "medium", "low") if lvl in by_conf]
    cols = st.columns(len(levels))
    for col, lvl in zip(cols, levels):
        rate, n = by_conf[lvl]
        col.metric(CONFIDENCE_STYLES[lvl][0], _pct(rate), help=f"{n} settled predictions")
    st.caption(
        "Should read high > medium > low. Only predictions logged since "
        "confidence levels were added are counted."
    )


def _render_goals(report: Dict[str, Any]) -> None:
    goals = report.get("goals") or {}
    if not goals:
        return
    st.markdown("#### ⚽ Goals markets")
    labels = [
        ("over_1.5", "Over/Under 1.5"),
        ("over_2.5", "Over/Under 2.5"),
        ("over_3.5", "Over/Under 3.5"),
        ("btts", "Both teams score"),
    ]
    present = [(k, label) for k, label in labels if k in goals]
    cols = st.columns(len(present))
    for col, (key, label) in zip(cols, present):
        m = goals[key]
        col.metric(
            label,
            _pct(m["hit_rate"]),
            help=(
                f"Called correctly {_pct(m['hit_rate'])} of {m['n']} times "
                f"(the side above 50%); {_ci_text(m['hit_rate_ci'])}. "
                f"It happened in {_pct(m['base_rate'])} of matches. "
                f"Brier {m['brier']:.3f} (lower is better)."
            ),
        )
    st.caption(
        "Hit rate of calling yes/no correctly. Only predictions logged since "
        "goals markets were tracked are counted."
    )


def _render_market_benchmark(days: int) -> None:
    """Model vs bookmaker odds on the same settled fixtures — the honest
    yardstick. Pre-match odds (market average, football-data.co.uk) are
    the strongest public forecast; the gap shows how far the model is
    from it."""
    try:
        bm = odds.benchmark(days=days)
    except Exception:
        return
    if not bm.get("n"):
        return

    st.markdown("#### 🆚 Model vs the betting market")
    c1, c2 = st.columns(2)
    c1.metric(
        "Log-loss — this model",
        f"{bm['model_logloss']:.3f}",
        delta=f"{bm['model_logloss'] - bm['market_logloss']:+.3f} vs market",
        delta_color="inverse",
        help="Lower is better. Model probabilities before any market blend "
        "— the honest comparison. Positive delta means the market priced "
        "these fixtures better than the model did.",
    )
    c2.metric(
        "Log-loss — bookmaker odds",
        f"{bm['market_logloss']:.3f}",
        help="Implied probabilities from pre-match market-average odds "
        "(football-data.co.uk), margin removed.",
    )
    st.caption(
        f"Scored on {bm['n']} of {bm['total']} settled fixtures with "
        f"bookmaker odds available. Model hit rate {bm['model_hit'] * 100:.0f}% "
        f"vs market favourite {bm['market_hit'] * 100:.0f}%."
    )
    if bm.get("blend_n"):
        diff = bm["blend_base_logloss"] - bm["blend_logloss"]
        direction = "improved" if diff > 0 else "worsened"
        st.caption(
            f"Market odds were blended into {bm['blend_n']} of these predictions: "
            f"log-loss {bm['blend_base_logloss']:.3f} unblended → "
            f"{bm['blend_logloss']:.3f} blended ({direction} by {abs(diff):.3f})."
        )


def history_rows(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Table rows for the history view, one per logged prediction."""
    rows = []
    for e in entries:
        probs = {k: float(e.get(f) or 0.0) for k, f in
                 (("home_win", "p_home"), ("draw", "p_draw"), ("away_win", "p_away"))}
        pick = max(probs, key=probs.get)
        if e.get("result"):
            score = f"{e.get('home_score')}-{e.get('away_score')}"
            verdict = "✅" if e.get("correct") else "❌"
        else:
            score, verdict = "—", "⏳"
        over = e.get("p_over25")
        btts = e.get("p_btts")
        rows.append({
            "Kickoff": (e.get("date") or "")[:16].replace("T", " "),
            "League": league_name(e.get("league")),
            "Match": f"{e.get('home_team')} vs {e.get('away_team')}",
            "Home / Draw / Away": " / ".join(_pct(probs[k]) for k in _PICK_LABELS),
            "Pick": _PICK_LABELS[pick],
            "Confidence": (e.get("confidence") or "—").capitalize(),
            "Predicted score": e.get("predicted_score") or "—",
            "Result": score,
            "Correct": verdict,
            "Over 2.5": _pct(over) if over is not None else "—",
            "BTTS": _pct(btts) if btts is not None else "—",
        })
    return rows


def _render_history(days: int, service: LeagueService) -> None:
    st.markdown("### 🗂️ Prediction history")
    scope = st.radio(
        "Show",
        ["This league", "All leagues"],
        horizontal=True,
        key="track_scope",
    )
    league = service.get_current_code() if scope == "This league" and service else None
    entries = prediction_log.history(days=days, league=league)
    if not entries:
        st.caption("No predictions logged for this period yet.")
        return
    # Settled first: "predict all" on the fixture pages logs fixtures
    # months ahead, which would otherwise crowd the top of the list.
    status = st.radio(
        "Status",
        ["Settled", "Awaiting result", "All"],
        horizontal=True,
        key="track_status",
    )
    if status == "Settled":
        entries = [e for e in entries if e.get("result")]
    elif status == "Awaiting result":
        # Soonest kickoff first — that's the one about to be scored.
        entries = [e for e in entries if not e.get("result")][::-1]
    if not entries:
        st.caption("Nothing matches this filter.")
        return
    st.dataframe(pd.DataFrame(history_rows(entries)), hide_index=True, use_container_width=True)
    st.caption(
        "Each row is the last prediction made before kickoff. ⏳ = not played "
        "or result not in yet. Kickoffs in SAST."
    )
