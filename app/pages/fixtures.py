"""Fixtures page with filters and per-match predictions."""

from datetime import date, datetime
from itertools import groupby
from typing import Any, Dict, List

import streamlit as st

from app.ui import (
    apply_theme,
    display_disclaimer,
    display_header,
    probability_bar,
    show_warning_message,
    COLORS,
)
from src.data import LeagueService


def _parse_date(m: Dict[str, Any]) -> date:
    try:
        return datetime.fromisoformat(m["date"]).date()
    except Exception:
        return datetime.min.date()


def render(service: LeagueService):
    apply_theme()
    display_header(service.get_current_name(), "Full fixture list")

    st.markdown("## 📅 Fixtures")
    matches = service.get_matches()
    if not matches:
        show_warning_message("No fixtures available for this league.")
        return

    if "fixture_predictions" not in st.session_state:
        st.session_state["fixture_predictions"] = {}

    _render_filters(matches, service)
    filtered = _apply_filters(matches)

    if not filtered:
        st.info("No fixtures match the selected filters.")
        return

    for date_label, day_matches in groupby(filtered, key=lambda m: _date_label(m)):
        st.markdown(f"### 📆 {date_label}")
        for m in day_matches:
            _render_fixture(service, m)
        st.markdown("---")

    display_disclaimer()


def _date_label(m: Dict[str, Any]) -> str:
    try:
        return datetime.fromisoformat(m["date"]).strftime("%a, %d %b %Y")
    except Exception:
        return "TBD"


def _render_filters(matches: List[Dict[str, Any]], service: LeagueService):
    rounds = sorted({m.get("round", "") for m in matches if m.get("round")})
    dates = sorted({_parse_date(m) for m in matches if m.get("date")})

    if st.button("🔮 Predict All Displayed", type="primary", use_container_width=True):
        predictions = st.session_state["fixture_predictions"]
        to_predict = [m for m in _apply_filters(matches) if m.get("status") == "scheduled"]
        for m in to_predict:
            pred = service.predict(m["home_team"], m["away_team"])
            pred["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            predictions[m["match_id"]] = pred
        st.success(f"Predicted {len(to_predict)} displayed matches")
        st.rerun()

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        options = ["All rounds"] + rounds
        st.selectbox("🏟️ Round", options, key="fixtures_round")
    with c2:
        if dates:
            today = date.today()
            default_date = next((d for d in dates if d >= today), dates[-1])
            st.date_input(
                "📅 Date",
                value=default_date,
                min_value=dates[0],
                max_value=dates[-1],
                key="fixtures_date",
            )
    with c3:
        st.checkbox("🕓 Show completed", value=False, key="fixtures_show_completed")

    if st.session_state["fixture_predictions"]:
        if st.button("🗑️ Clear predictions", use_container_width=True):
            st.session_state["fixture_predictions"] = {}
            st.rerun()


def _apply_filters(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    round_filter = st.session_state.get("fixtures_round", "All rounds")
    date_filter = st.session_state.get("fixtures_date", None)
    show_completed = st.session_state.get("fixtures_show_completed", False)

    filtered = matches
    if round_filter != "All rounds":
        filtered = [m for m in filtered if m.get("round") == round_filter]
    if date_filter:
        try:
            d = date_filter if isinstance(date_filter, date) else date_filter.date()
            filtered = [m for m in filtered if _parse_date(m) == d]
        except Exception:
            pass

    upcoming = [m for m in filtered if m.get("status") == "scheduled"]
    completed = [m for m in filtered if m.get("status") == "full_time"]
    other = [m for m in filtered if m.get("status") not in ("scheduled", "full_time")]

    upcoming.sort(key=lambda m: m.get("date") or "")
    completed.sort(key=lambda m: m.get("date") or "", reverse=True)
    other.sort(key=lambda m: m.get("date") or "")

    result = upcoming + other
    if show_completed:
        result = completed + result
    return result


def _render_fixture(service: LeagueService, m: Dict[str, Any]):
    fixture_id = m["match_id"]
    predictions = st.session_state["fixture_predictions"]
    home = m["home_team"]
    away = m["away_team"]

    c_info, c_match, c_action = st.columns([1.4, 2.6, 1.2])

    with c_info:
        try:
            dt = datetime.fromisoformat(m["date"]) if m.get("date") else None
            time = dt.strftime("%H:%M") + " SAST" if dt else "—"
        except Exception:
            time = "—"
        st.markdown(
            f"""**{time}**  
<span style='color:#a0a0a0; font-size:0.8rem;'>{m.get('round', '')}</span>""",
            unsafe_allow_html=True,
        )

    with c_match:
        if m.get("status") == "full_time" and m.get("home_score") is not None:
            st.markdown(f"{home} {m['home_score']}-{m['away_score']} {away}")
        else:
            st.markdown(f"{home} vs {away}")
        st.caption(f"Source: {m.get('source', 'unknown')}")

    with c_action:
        if m.get("status") == "full_time":
            st.caption("Completed")
        elif st.button("🔮 Predict", key=f"predict_{fixture_id}", use_container_width=True):
            pred = service.predict(home, away)
            pred["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            predictions[fixture_id] = pred
            st.rerun()

    if fixture_id in predictions:
        _render_prediction_summary(predictions[fixture_id], home, away)


def _render_prediction_summary(pred: Dict[str, Any], home: str, away: str):
    outcome = pred.get("outcome_probabilities", {})
    score = pred.get("predicted_score", {})
    with st.container():
        c_score, c_bars = st.columns([1, 2])
        with c_score:
            st.markdown(
                f"""
                <div style="text-align:center; padding:0.5rem;">
                    <div style="font-size:0.75rem; color:#a0a0a0;">Predicted Score</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#00ff85;">{score.get('score', '—')}</div>
                    <div style="font-size:0.7rem; color:#a0a0a0;">{pred.get('model', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_bars:
            probability_bar(f"🏠 {home}", outcome.get("home_win", 0), COLORS["home_win"])
            probability_bar("🤝 Draw", outcome.get("draw", 0), COLORS["draw"])
            probability_bar(f"✈️ {away}", outcome.get("away_win", 0), COLORS["away_win"])
    if pred.get("predicted_at"):
        st.caption(f"Predicted on {pred['predicted_at']}")
