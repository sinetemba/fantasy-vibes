"""Fixtures page with filters and per-match predictions."""

from datetime import date, datetime, timedelta
from itertools import groupby
from typing import Any, Dict, List

MAX_BATCH_PREDICTIONS = 20

import streamlit as st

from app.ui import (
    apply_theme,
    display_disclaimer,
    display_header,
    form_badges_html,
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

    filtered = _render_filters(matches, service)

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
    current_code = service.get_current_code()
    if st.session_state.get("fixtures_league") != current_code:
        st.session_state["fixtures_league"] = current_code
        st.session_state.pop("fixtures_date", None)
        st.session_state.pop("fixtures_round", None)

    rounds = sorted({m.get("round", "") for m in matches if m.get("round")})
    dates = sorted({_parse_date(m) for m in matches if m.get("date")})

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        options = ["All rounds"] + rounds
        if "fixtures_round" in st.session_state and st.session_state["fixtures_round"] not in options:
            st.session_state["fixtures_round"] = "All rounds"
        st.selectbox("🏟️ Round", options, key="fixtures_round")
    with c2:
        st.checkbox("📅 Filter by date", value=False, key="fixtures_use_date")
        if dates and st.session_state.get("fixtures_use_date"):
            today = date.today()
            default_date = today
            min_value = min(today, dates[0])
            max_value = max(today, dates[-1])
            if "fixtures_date" in st.session_state:
                stored = st.session_state["fixtures_date"]
                try:
                    stored_date = stored if isinstance(stored, date) else stored.date()
                    if not (min_value <= stored_date <= max_value):
                        st.session_state["fixtures_date"] = default_date
                except Exception:
                    st.session_state["fixtures_date"] = default_date
            st.date_input(
                "Date",
                value=default_date,
                min_value=min_value,
                max_value=max_value,
                key="fixtures_date",
                label_visibility="collapsed",
            )
    with c3:
        st.checkbox("🗓️ This week's fixtures", value=False, key="fixtures_this_week")
        st.checkbox("🔮 Show all upcoming", value=False, key="fixtures_show_all_upcoming")
        st.checkbox("🕓 Show completed", value=False, key="fixtures_show_completed")

    filtered = _apply_filters(matches)

    if filtered:
        if st.button(f"🔮 Predict first {MAX_BATCH_PREDICTIONS} scheduled", type="primary", use_container_width=True):
            predictions = st.session_state["fixture_predictions"]
            all_scheduled = [m for m in filtered if m.get("status") == "scheduled"]
            to_predict = all_scheduled[:MAX_BATCH_PREDICTIONS]
            for m in to_predict:
                pred = service.predict(m["home_team"], m["away_team"])
                pred["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                predictions[m["match_id"]] = pred
            msg = f"Predicted {len(to_predict)} displayed matches"
            if len(all_scheduled) > MAX_BATCH_PREDICTIONS:
                msg += f" (capped at {MAX_BATCH_PREDICTIONS})"
            st.success(msg)

    if st.session_state["fixture_predictions"]:
        if st.button("🗑️ Clear predictions", use_container_width=True):
            st.session_state["fixture_predictions"] = {}
            st.rerun()

    return filtered


def _apply_filters(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    round_filter = st.session_state.get("fixtures_round", "All rounds")
    use_date = st.session_state.get("fixtures_use_date", False)
    date_filter = st.session_state.get("fixtures_date") if use_date else None
    show_completed = st.session_state.get("fixtures_show_completed", False)
    show_all_upcoming = st.session_state.get("fixtures_show_all_upcoming", False)
    this_week = st.session_state.get("fixtures_this_week", False)

    filtered = matches
    if round_filter != "All rounds":
        filtered = [m for m in filtered if m.get("round") == round_filter]

    if this_week:
        # Everything in the 7-day window starting today, regardless of status.
        start = date.today()
        end = start + timedelta(days=6)
        filtered = [
            m for m in filtered
            if m.get("date") and start <= _parse_date(m) <= end
        ]
        filtered.sort(key=lambda m: m.get("date") or "")
        return filtered

    if date_filter:
        try:
            d = date_filter if isinstance(date_filter, date) else date_filter.date()
            filtered = [m for m in filtered if _parse_date(m) == d]
        except Exception:
            pass

    if show_all_upcoming:
        upcoming = [m for m in filtered if m.get("status") == "scheduled"]
        upcoming.sort(key=lambda m: m.get("date") or "")
        return upcoming

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
        if m.get("time_confirmed") is False:
            time = "TBC"
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
        _render_prediction_summary(predictions[fixture_id], home, away, service=service)


def _render_prediction_summary(
    pred: Dict[str, Any], home: str, away: str, service: LeagueService = None
):
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
            home_win = outcome.get("home_win", 0)
            away_win = outcome.get("away_win", 0)
            draw = outcome.get("draw", 0)
            if home_win >= away_win:
                home_color, away_color = COLORS["home_win"], COLORS["away_win"]
            else:
                home_color, away_color = COLORS["away_win"], COLORS["home_win"]
            probability_bar(f"🏠 {home}", home_win, home_color)
            probability_bar("🤝 Draw", draw, COLORS["draw"])
            probability_bar(f"✈️ {away}", away_win, away_color)
    if service is not None:
        st.markdown(
            f"<div style='font-size:0.8rem;color:#a0a0a0;margin-top:0.3rem;'>"
            f"🏠 Form: {form_badges_html(service.get_team_form(home))}"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"✈️ Form: {form_badges_html(service.get_team_form(away))}"
            f"&nbsp;<span style='font-size:0.7rem;'>(latest →)</span></div>",
            unsafe_allow_html=True,
        )
    factors = _format_model_inputs(pred, home, away)
    if factors:
        st.caption(factors)
    if pred.get("predicted_at"):
        st.caption(f"Predicted on {pred['predicted_at']}")


def _format_model_inputs(pred: Dict[str, Any], home: str, away: str) -> str:
    """Summarise the form/log-position factors behind a prediction."""
    inputs = pred.get("model_inputs") or {}

    def _fmt(team: str, info: Dict[str, Any]) -> str:
        parts = []
        pos, league = info.get("position"), info.get("context_league")
        if pos and league:
            parts.append(f"#{pos} in {league}")
        if info.get("form_ppg") is not None:
            parts.append(f"{info['form_ppg']:.1f} pts/game")
        comp_ppg = info.get("comp_form_ppg")
        comp = info.get("competition")
        if comp_ppg is not None and comp and comp != league:
            parts.append(f"{comp_ppg:.1f} in {comp}")
        return f"{team}: " + " · ".join(parts) if parts else ""

    bits = [_fmt(home, inputs.get("home") or {}), _fmt(away, inputs.get("away") or {})]
    return "  |  ".join(b for b in bits if b)
